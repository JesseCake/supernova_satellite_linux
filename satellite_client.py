import os
import socket
import struct
import threading
import queue
import time
from typing import Tuple, Optional

import numpy as np
import pyaudio
import pvporcupine
import yaml

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
SPK_RATE = 44100
SAMPLE_WIDTH = 2  # int16
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1         # <<< stereo out
FORMAT = pyaudio.paInt16
CHUNK = 1024  # streaming chunk to server
OUTPUT_CHUNK = 4096  # playback chunk size

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
                frames_per_buffer=1024,
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
        return self._in_stream.read(n_frames, exception_on_overflow=False)

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
        self.queue: "queue.Queue[bytes]" = queue.Queue(maxsize=128)
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
        idle_grace = 0.15
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
        porcupine_access_key: str,
        keyword_paths: Optional[list] = None,
        built_in_keywords: Optional[list] = None,
        wake_sensitivity: float = 0.6,
        silence_ms: int = 800,
        level_threshold: int = 150,  # ~ RMS threshold to detect speech (int16)
        audio_cfg=None,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.silence_ms = silence_ms
        self.level_threshold = level_threshold

        # Audio
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

        # Porcupine (wake word)
        # Either provide custom .ppn paths via keyword_paths OR use built-in keyword names
        if keyword_paths:
            self.porcupine = pvporcupine.create(
                access_key=porcupine_access_key,
                keyword_paths=keyword_paths,
                sensitivities=[wake_sensitivity] * len(keyword_paths),
            )
        else:
            if not built_in_keywords:
                built_in_keywords = ["porcupine"]  # default built-in
            self.porcupine = pvporcupine.create(
                access_key=porcupine_access_key,
                keywords=built_in_keywords,
                sensitivities=[wake_sensitivity] * len(built_in_keywords),
            )

        # For wake detection, Porcupine tells us what frame length to read
        self.frame_len = self.porcupine.frame_length

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
                    # int16 mono 16k — play immediately
                    self.playback.enqueue(payload)
                elif tag == b"BEEP":
                    print(f"[recv] BEEP {len(payload)} bytes")
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

    def _stream_utterance(self):
        """
        After RDY0, stream microphone audio as AUD0 in chunks until local silence timeout.
        We rely on the server's VAD, but sending STOP helps flush faster.
        """
        # Use a bigger chunk for network efficiency during streaming
        self.audio.open_input(frames_per_buffer=CHUNK)

        silence_start = None

        # Start streaming loop
        while True:
            pcm = self.audio.read_input(CHUNK)  # int16 little-endian

            # If we are speaking (playing TTS/beeps), do NOT send audio.
            if self.speaking_event.is_set():
                # Optional: treat as silence for the local-stop timer
                silence_start = None
                time.sleep(0.01)  # yield; avoid busy loop
                continue

            # Send chunk
            try:
                self._send(b"AUD0", pcm)
            except Exception:
                break

            # crude silence gate to decide when to stop
            if self._rms_int16(pcm) < self.level_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif (time.time() - silence_start) * 1000 >= self.silence_ms:
                    # Enough silence — end utterance
                    try:
                        self._send(b"STOP")
                    except Exception:
                        pass
                    break
            else:
                silence_start = None

    def wake_and_talk_once(self):
        """
        Blocks until wake word is detected, performs WAKE handshake,
        waits for RDY0, streams an utterance, and then returns.
        """
        # Wake-word listening stream
        self.audio.open_input(frames_per_buffer=self.frame_len)

        #dbg = 0
        print("[satellite] Listening for wake word...")
        while True:
            pcm = self.audio.read_input(self.frame_len)
            #rms = self._rms_int16(pcm)
            #dbg += 1
            #if dbg % 50 == 0:
            #    print(f"[debug] mic RMS: {rms:.1f}")

            # Porcupine expects int16 little-endian mono
            frame = np.frombuffer(pcm, dtype=np.int16)
            result = self.porcupine.process(frame)
            if result >= 0:
                print("[satellite] Wake detected!")
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
        # Stream a single utterance
        self._stream_utterance()

        # Keep the connection open to receive TTS0 reply frames.
        # We give it a short window; receiver thread will keep playing any incoming audio.
        time.sleep(2.0)

        # Done with this interaction; close socket and go back to wake loop
        self.close()
        print("[satellite] Interaction finished; back to wake listening.")

    def run_forever(self):
        try:
            while True:
                self.wake_and_talk_once()
        except KeyboardInterrupt:
            print("\n[satellite] Stopping...")
        finally:
            self.playback.stop()
            self.audio.close()
            try:
                self.porcupine.delete()
            except Exception:
                pass

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
        porcupine_access_key=cfg["pv_access_key"],
        keyword_paths=cfg["wake_word"].get("keyword_paths"),
        wake_sensitivity=cfg["wake_word"].get("sensitivity", 0.6),
        silence_ms=800,
        level_threshold=150,
    )
    client.run_forever()
