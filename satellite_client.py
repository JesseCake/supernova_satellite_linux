import os
import socket
import struct
import threading
import queue
import time
from typing import Tuple, Optional

import numpy as np
import pyaudio
#from openwakeword.model import Model as OWWModel
import yaml

import openwakeword

from pathlib import Path
import importlib.resources as ir

def oww_resources_ok() -> bool:
    # openWakeWord stores these under openwakeword/resources/models/
    with ir.as_file(ir.files("openwakeword") / "resources" / "models") as models_dir:
        models_dir = Path(models_dir)
        required = [
            "melspectrogram.onnx",
            # Depending on version/framework, you may also need an embedding model.
            # If you want to be stricter, add "embedding_model.onnx" here if present in your release.
        ]
        return all((models_dir / f).is_file() for f in required)

def ensure_oww_resources():
    if oww_resources_ok():
        return
    import openwakeword.utils
    openwakeword.utils.download_models()

# -------------------------
# Connection / framing
# -------------------------
FRAME_HDR = struct.Struct("<4sI")  # (type:4s, length:uint32)

def pack_frame(tag: bytes, payload: bytes = b"") -> bytes:
    return FRAME_HDR.pack(tag, len(payload)) + payload

def read_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)

def read_frame(sock: socket.socket) -> Tuple[bytes, bytes]:
    header = read_exactly(sock, FRAME_HDR.size)
    tag, length = FRAME_HDR.unpack(header)
    payload = read_exactly(sock, length) if length else b""
    return tag, payload

# -------------------------
# Audio IO (PyAudio)
# -------------------------
MIC_RATE = 16000
SPK_RATE = 16000
SAMPLE_WIDTH = 2  # int16
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1         # <<< stereo out
FORMAT = pyaudio.paInt16
CHUNK = 1024  # streaming chunk to server
OUTPUT_CHUNK = 4096  # playback chunk size

# Automatic mic input gain applied after every read_input() call.
# Affects both wake-word detection and streamed AUD0 frames sent to the server.
#
# INPUT_GAIN_TARGET_RMS: desired loudness on a 0.0–1.0 float32 scale.
#   Raise if wake word still misses or Whisper struggles; lower if over-driven.
# INPUT_GAIN_MAX_DB: hard ceiling on gain so silence/noise isn't boosted to a roar.
# INPUT_GAIN_NOISE_FLOOR: frames quieter than this RMS are left untouched
#   (prevents amplifying breath noise / silence between words).
INPUT_GAIN_TARGET_RMS  = 0.2
INPUT_GAIN_MAX_DB      = 20.0
INPUT_GAIN_NOISE_FLOOR = 0.005


def list_devices():
    p = pyaudio.PyAudio()
    devs = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        devs.append((i, info["name"], info["maxInputChannels"], info["maxOutputChannels"]))
    p.terminate()
    return devs

def find_device_indices_by_name(pa, input_name=None, output_name=None):
    in_idx = out_idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info["name"]
        max_in = int(info["maxInputChannels"])
        max_out = int(info["maxOutputChannels"])
        if input_name and input_name in name and max_in > 0:
            in_idx = i
        if output_name and output_name in name and max_out > 0:
            out_idx = i
    return in_idx, out_idx

class AudioIO:
    def __init__(self):
        self.p = pyaudio.PyAudio()

        # List devices (diagnostic)
        #print("=== PyAudio devices ===")
        #for i, name, ch_in, ch_out in list_devices():
        #    print(f"{i:2d} | {name} | in:{ch_in} out:{ch_out}")

        # Open OUTPUT explicitly on pulse (fallback to default if pulse missing)
        try:
            self.out = self.p.open(
                format=pyaudio.paInt16,
                channels=OUTPUT_CHANNELS,
                rate=SPK_RATE,  # 16k; Pulse will resample if needed
                output=True,
                frames_per_buffer=OUTPUT_CHUNK,
                start=False,
            )
            self.out.start_stream()
            print("[audio] Output started. active:", self.out.is_active(), "stopped:", self.out.is_stopped())
        except Exception as e:
            print("[audio] FAILED to open/start output stream:", e)
            raise

        self._in_stream = None

    def open_input(self, frames_per_buffer: int):
        # Reopen INPUT explicitly on pulse (fallback to default if pulse missing)
        if self._in_stream:
            try:
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        try:
            self._in_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=MIC_RATE,
                input=True,
                frames_per_buffer=frames_per_buffer,
                #input_device_index=0,  # may be None (default)
                start=False,
            )
            self._in_stream.start_stream()
        except Exception as e:
            print("[audio] FAILED to open/start input stream:", e)
            raise
    
    def pause_input(self):
        """Pause mic capture (idempotent)."""
        if self._in_stream and not self._in_stream.is_stopped():
            try:
                self._in_stream.stop_stream()
                # print("[audio] input paused")
            except Exception:
                pass
    def resume_input(self):
        """Resume mic capture (idempotent)."""
        if self._in_stream and self._in_stream.is_stopped():
            try:
                self._in_stream.start_stream()
                # print("[audio] input resumed")
            except Exception:
                pass

    def read_input(self, n_frames: int) -> bytes:
        """Block while paused so callers don't read from a stopped stream."""
        while self._in_stream and self._in_stream.is_stopped():
            time.sleep(0.005)
        raw = self._in_stream.read(n_frames, exception_on_overflow=False)

        # Auto-gain: normalise frame to TARGET_RMS, capped at MAX_DB, noise floor gated
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms >= INPUT_GAIN_NOISE_FLOOR:
            max_gain = 10.0 ** (INPUT_GAIN_MAX_DB / 20.0)
            gain = min(INPUT_GAIN_TARGET_RMS / rms, max_gain)
            arr = np.clip(arr * gain, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    def play_bytes(self, pcm_bytes: bytes):
            self.out.write(pcm_bytes)

    def beep(self, freq=800.0, duration=0.5, volume=0.5, rate=16000):
        n = int(rate * duration)
        t = np.arange(n, dtype=np.float32) / rate
        wave = np.sin(2 * np.pi * freq * t) * volume
        pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        print(f"[audio] Beep: {freq}Hz/{duration}s → {len(pcm)} bytes")
        self.play_bytes(pcm)

    def close(self):
        try:
            if self._in_stream:
                self._in_stream.close()
            if self.out:
                self.out.close()
            self.p.terminate()
        except Exception:
            pass


# -------------------------
# Playback worker
# -------------------------
class PlaybackThread(threading.Thread):
    def __init__(self, audio: AudioIO, speaking_event: threading.Event):
        super().__init__(daemon=True)
        self.audio = audio
        self.queue: "queue.Queue[bytes]" = queue.Queue(maxsize=-1) # unbounded
        self._stop = threading.Event()
        self._speaking = False
        self._speaking_event = speaking_event


    def _set_speaking(self, val: bool):
        if val != self._speaking:
            self._speaking = val
            if val:
                self._speaking_event.set()
            else:
                self._speaking_event.clear()

    def run(self):
        idle_grace = 0.12
        last_write_t = 0.0
        while not self._stop.is_set():
            try:
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                if self._speaking and (time.time() - last_write_t) >= idle_grace and self.queue.empty():
                    self._set_speaking(False)
                continue

            if data is None:
                break
            if not self._speaking:
                self._set_speaking(True)

            self.audio.play_bytes(data)
            last_write_t = time.time()

        self._set_speaking(False)

    def enqueue(self, data: bytes):
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            # drop if overloaded
            pass

    def stop(self):
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

# -------------------------
# Satellite client
# -------------------------
class SatelliteClient:
    def __init__(
        self,
        server_host: str,
        server_port: int,
        wake_cfg: dict,
        silence_ms: int = 800,
        level_threshold: int = 150,  # ~ RMS threshold to detect speech (int16)
        audio_cfg=None,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.silence_ms = silence_ms
        self.level_threshold = level_threshold
        self.wake_cfg = wake_cfg

        # Audio device setup
        self.audio_cfg = audio_cfg or {}

        self.audio = AudioIO()
        self.speaking_event = threading.Event()
        self.playback = PlaybackThread(self.audio, speaking_event=self.speaking_event)
        self.playback.start()

        print("[satellite] Playing startup beeps…")
        self.audio.beep(800, 0.12, 0.2)   # high beep
        time.sleep(0.08)
        self.audio.beep(400, 0.12, 0.2)   # low beep

        # Open Wake Word
        model_path = self.wake_cfg.get("model_path", "./soopanova.onnx")
        self.wake_threshold = float(self.wake_cfg.get("threshold", 0.05))
        self.wake_cooldown_s = float(self.wake_cfg.get("cooldown_s", 0.8))
        self.post_session_silence_s = 0.4      # ringdown after playback stops

        # OWW params
        self.oww_hop = 1280
        # Ensure resources are present
        ensure_oww_resources()
        # Load model
        self.oww = openwakeword.Model(wakeword_model_paths=[model_path])

        # Networking state
        self.sock: Optional[socket.socket] = None
        self._receiver = None
        self._ready_event = threading.Event()  # set after RDY0

    # ---------- networking ----------
    def connect(self):
        if self.sock:
            self.sock.close()
        self.sock = socket.create_connection((self.server_host, self.server_port))
        # Start receiver thread
        self._receiver = threading.Thread(target=self._recv_loop, daemon=True)
        self._receiver.start()

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _recv_loop(self):
        try:
            while True:
                tag, payload = read_frame(self.sock)
                if tag == b"TTS0":
                    print(f"[recv] TTS0 {len(payload)} bytes")
                    # Pre-mark speaking to block mic immediately
                    self.speaking_event.set()
                    # int16 mono 16k — play immediately
                    self.playback.enqueue(payload)
                elif tag == b"BEEP":
                    print(f"[recv] BEEP {len(payload)} bytes")
                    self.speaking_event.set()
                    self.playback.enqueue(payload)
                elif tag == b"RDY0":
                    print("[recv] RDY0")
                    self._ready_event.set()
                elif tag == b"CLOS":
                    print("[recv] CLOS")
                    # server closing channel
                    self._ready_event.clear()
                else:
                    # ignore unknown tags
                    print(f"[recv] Unknown tag {tag!r} ({len(payload)} bytes)")
        except Exception as e:
            # socket closed or error — leave
            print(f"[recv] receiver exiting: {e}")
            self._ready_event.clear()

    # ---------- satellite flow ----------
    def _send(self, tag: bytes, payload: bytes = b""):
        self.sock.sendall(pack_frame(tag, payload))

    def _rms_int16(self, pcm_bytes: bytes) -> float:
        # quick-n-dirty speech activity metric
        if not pcm_bytes:
            return 0.0
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))

    def _flush_wake_model(self, seconds: float = 2.0):
        """
        openWakeWord keeps internal state/buffer across predict() calls.
        After a wake/session, flush it with silence so we don't immediately retrigger.
        """
        # If a reset method exists in your version, use it.
        for name in ("reset", "reset_states", "reset_state"):
            fn = getattr(self.oww, name, None)
            if callable(fn):
                fn()
                return

        # Otherwise, push silence through predict() to flush internal buffers.
        n_samples = int(MIC_RATE * seconds)              # e.g. 2.0s -> 32000 samples
        step = self.oww_hop                             # 1280 samples per frame (80ms)
        silence = np.zeros(step, dtype=np.int16)

        for _ in range((n_samples + step - 1) // step):
            _ = self.oww.predict(silence)

    
    def _wait_playback_drain(self, timeout_s: float = 5.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if (not self.speaking_event.is_set()) and self.playback.queue.empty():
                return True
            time.sleep(0.02)
        return False

    def _flush_mic(self, n_frames: int, count: int):
        # Read and discard a few frames to clear ALSA/Pulse buffers
        for _ in range(count):
            try:
                _ = self.audio.read_input(n_frames)
            except Exception:
                break


    def _stream_audio_session(self):
        """
        After RDY0, continuously stream mic audio as AUD0 until the server ends
        the session (CLOS clears _ready_event) or the socket dies.
        While the server is speaking (TTS/beeps), don't send mic audio.
        """
        self.audio.open_input(frames_per_buffer=CHUNK)
        was_speaking = False

        while self._ready_event.is_set():
            # If server is speaking, pause mic and wait
            if self.speaking_event.is_set():
                if not was_speaking:
                    self.audio.pause_input()
                    was_speaking = True
                time.sleep(0.01)
                continue
            else:
                if was_speaking:
                    self.audio.resume_input()
                    was_speaking = False

            try:
                pcm = self.audio.read_input(CHUNK)
            except Exception:
                break

            try:
                self._send(b"AUD0", pcm)
            except Exception:
                break

    def talk_once_no_wake(self):
        """Start a session immediately (no Porcupine)."""
        self.connect()
        self._ready_event.clear()

        try:
            self._send(b"WAKE")
        except Exception as e:
            print(f"[satellite] Error sending WAKE: {e}")
            self.close()
            return

        print("[satellite] Waiting for RDY0 from server...")
        if not self._ready_event.wait(timeout=5.0):
            print("[satellite] Timed out waiting for RDY0.")
            self.close()
            return

        print("[satellite] RDY0 received. Start talking.")
        self._stream_audio_session()

        # Wait for local playback to fully drain before closing
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if not self.speaking_event.is_set() and self.playback.queue.empty():
                break
            time.sleep(0.05)

        self.close()
        print("[satellite] Interaction finished.")

        
    def wake_and_talk_once(self):
        """
        Blocks until wake word is detected, performs WAKE handshake,
        waits for RDY0, streams an utterance, and then returns.
        """
        # Wake-word listening stream
        self.audio.open_input(frames_per_buffer=self.oww_hop)

        # Discard a few frames (clears buffered tail of TTS/beeps)
        self._flush_mic(self.oww_hop, 3)


        print("[satellite] Listening for wake word...")
        
        next_allowed_wake_t = 0.0

        while True:
            pcm = self.audio.read_input(self.oww_hop)
            frame = np.frombuffer(pcm, dtype=np.int16)

            pred = self.oww.predict(frame)
            if not pred:
                continue

            model_name = max(pred, key=pred.get)
            score = float(pred[model_name])
            now = time.time()

            # simple time-based cooldown only
            if now < next_allowed_wake_t:
                continue

            if score >= self.wake_threshold:
                print(f"[satellite] Wake word '{model_name}' detected (score={score:.3f})")
                next_allowed_wake_t = now + self.wake_cooldown_s
                break

        # Connect (or reconnect) to the server and send WAKE
        self.connect()
        self._ready_event.clear()
        try:
            self._send(b"WAKE")
        except Exception as e:
            print(f"[satellite] Error sending WAKE: {e}")
            self.close()
            return

        # Wait for RDY0 (server says “I’m here” via TTS0 while we wait)
        print("[satellite] Waiting for RDY0 from server...")
        if not self._ready_event.wait(timeout=5.0):
            print("[satellite] Timed out waiting for RDY0.")
            self.close()
            return

        print("[satellite] RDY0 received. Start talking.")
        # Stream until server ends the session (CLOS) or socket error
        self._stream_audio_session()

        # Wait for local playback to fully drain before closing
        deadline = time.time() + 30.0  # safety cap
        while time.time() < deadline:
            if not self.speaking_event.is_set() and self.playback.queue.empty():
                break
            time.sleep(0.05)

        time.sleep(self.post_session_silence_s)

        # Flush model state before returning to wake listening
        self._flush_wake_model(seconds=2.0)


        # Done with this interaction; close socket and go back to wake loop
        self.close()
        print("[satellite] Interaction finished; back to wake listening.")

    def run_forever(self):
        try:
            while True:
                ###########
                # Choose one of the following modes:
                ###########

                # Mode 1: No wake word; talk once immediately
                #self.talk_once_no_wake()
                # Mode 2: Wake word enabled; wait for wake then talk
                self.wake_and_talk_once()
        except KeyboardInterrupt:
            print("\n[satellite] Stopping...")
        finally:
            self.playback.stop()
            self.audio.close()

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":

    cfg = load_config()

    client = SatelliteClient(
        server_host=cfg["voice_server"]["host"],
        server_port=int(cfg["voice_server"]["port"]),
        wake_cfg=cfg.get("wake_word", {}),
    )
    client.run_forever()
