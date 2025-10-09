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
SPK_RATE = 16000
SAMPLE_WIDTH = 2  # int16
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1         # mono out
FORMAT = pyaudio.paInt16
CHUNK = 1024                # streaming chunk to server (must be multiple of Porcupine frame_len)
OUTPUT_CHUNK = 4096         # playback chunk size

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

class _InputRing:
    """Tiny ring buffer so one capture stream can feed multiple consumers."""
    def __init__(self, max_seconds=0.1, rate=16000, sample_width=2):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._max_bytes = int(max_seconds * rate * sample_width)

    def write(self, data: bytes):
        with self._cond:
            self._buf += data
            if len(self._buf) > self._max_bytes:
                self._buf = self._buf[-self._max_bytes:]
            self._cond.notify_all()

    def read_exact_frames(self, n_frames: int, sample_width=2) -> bytes:
        need = n_frames * sample_width
        with self._cond:
            while len(self._buf) < need:
                self._cond.wait(timeout=0.2)
            out = bytes(self._buf[:need])
            del self._buf[:need]
            return out

class AudioIO:
    def __init__(self):
        self.p = pyaudio.PyAudio()

        # Output
        try:
            self.out = self.p.open(
                format=pyaudio.paInt16,
                channels=OUTPUT_CHANNELS,
                rate=SPK_RATE,
                output=True,
                frames_per_buffer=OUTPUT_CHUNK,
                start=False,
            )
            self.out.start_stream()
            print("[audio] Output started. active:", self.out.is_active(), "stopped:", self.out.is_stopped())
        except Exception as e:
            print("[audio] FAILED to open/start output stream:", e)
            raise

        # Single capture path
        self._cap_stream = None
        self._cap_thread = None
        self._cap_stop = threading.Event()
        self._ring = _InputRing(max_seconds=0.1, rate=MIC_RATE, sample_width=SAMPLE_WIDTH)

    def _capture_loop(self, frames_per_buffer: int):
        while not self._cap_stop.is_set():
            try:
                data = self._cap_stream.read(frames_per_buffer, exception_on_overflow=False)
                self._ring.write(data)
            except Exception:
                time.sleep(0.005)

    def start_single_capture(self, frames_per_buffer: int, input_device_index: Optional[int] = None):
        """Open ONE input stream and start a background reader into a ring buffer."""
        if self._cap_stream:
            try:
                self._cap_stream.close()
            except Exception:
                pass
            self._cap_stream = None

        self._cap_stop.clear()
        self._cap_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=INPUT_CHANNELS,
            rate=MIC_RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=frames_per_buffer,
            start=True,
        )
        self._cap_thread = threading.Thread(
            target=self._capture_loop, args=(frames_per_buffer,), daemon=True
        )
        self._cap_thread.start()

    def read_frames(self, n_frames: int) -> bytes:
        """Blocking read of exactly n_frames from the ring (returns int16 bytes)."""
        return self._ring.read_exact_frames(n_frames, sample_width=SAMPLE_WIDTH)

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
            self._cap_stop.set()
            if self._cap_thread:
                self._cap_thread.join(timeout=0.2)
            if self._cap_stream:
                self._cap_stream.close()
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
        self.queue: "queue.Queue[bytes]" = queue.Queue(maxsize=-1)  # unbounded
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

    def flush(self):
        with self.queue.mutex:
            self.queue.queue.clear()

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
        level_threshold: int = 150,
        audio_cfg=None,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.silence_ms = silence_ms
        self.level_threshold = level_threshold

        # Audio
        self.audio_cfg = audio_cfg or {}
        self.audio = AudioIO()
        self.speaking_event = threading.Event()
        self.playback = PlaybackThread(self.audio, speaking_event=self.speaking_event)
        self.playback.start()

        print("[satellite] Playing startup beeps…")
        self.audio.beep(800, 0.12, 0.6)
        time.sleep(0.08)
        self.audio.beep(400, 0.12, 0.6)

        # Porcupine (wake word)
        if keyword_paths:
            self.porcupine = pvporcupine.create(
                access_key=porcupine_access_key,
                keyword_paths=keyword_paths,
                sensitivities=[wake_sensitivity] * len(keyword_paths),
            )
        else:
            if not built_in_keywords:
                built_in_keywords = ["porcupine"]
            self.porcupine = pvporcupine.create(
                access_key=porcupine_access_key,
                keywords=built_in_keywords,
                sensitivities=[wake_sensitivity] * len(built_in_keywords),
            )

        # Single input capture sized for Porcupine frame
        self.frame_len = self.porcupine.frame_length
        self.audio.start_single_capture(frames_per_buffer=self.frame_len)

        # Networking state
        self.sock: Optional[socket.socket] = None
        self._receiver = None
        self._ready_event = threading.Event()  # set after RDY0
        self._drop_tts_until_ready = False     # mute TTS after INT0 until RDY0

        # Background wake loop (optional if you want hotword during TTS)
        self._wake_enabled = threading.Event()
        self._wake_enabled.clear()  # start enabled (idle state)
        self._wake_thread = threading.Thread(target=self._wake_vad_loop, daemon=True)
        self._wake_thread.start()

    # ---------- networking ----------
    def connect(self):
        if self.sock:
            self.sock.close()
        self.sock = socket.create_connection((self.server_host, self.server_port))
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
                    if self._drop_tts_until_ready:
                        # print(f"[recv] TTS0 {len(payload)} bytes (dropped)")
                        continue
                    #print(f"[recv] TTS0 {len(payload)} bytes")
                    self._wake_enabled.set()  # allow wake during TTS

                    self.speaking_event.set()
                    self.playback.enqueue(payload)

                elif tag == b"BEEP":
                    if self._drop_tts_until_ready:
                        # print(f"[recv] BEEP {len(payload)} bytes (dropped)")
                        continue
                    self._wake_enabled.set()  # allow wake during TTS

                    self.speaking_event.set()
                    self.playback.enqueue(payload)

                elif tag == b"RDY0":
                    print("[recv] RDY0")
                    self._drop_tts_until_ready = False
                    self._wake_enabled.clear()  # disable wake during user speech
                    self._ready_event.set()

                elif tag == b"CLOS":
                    print("[recv] CLOS")
                    self._wake_enabled.set()
                    self._ready_event.clear()

                else:
                    print(f"[recv] Unknown tag {tag!r} ({len(payload)} bytes)")
        except Exception as e:
            print(f"[recv] receiver exiting: {e}")
            self._ready_event.clear()

    # ---------- satellite flow ----------
    def _send(self, tag: bytes, payload: bytes = b""):
        # print(f"[send] {tag!r} {len(payload)} bytes")
        self.sock.sendall(pack_frame(tag, payload))

    def _stream_audio_session(self):
        """
        After RDY0, continuously stream mic audio as AUD0 until the server ends
        the session (CLOS clears _ready_event) or the socket dies.
        While speaking, don't send mic audio.
        """
        while self._ready_event.is_set():
            #print("[debug] _ready_event is set, streaming audio...")
            if self.speaking_event.is_set():
                #print("[debug] speaking_event is set, not sending audio")
                time.sleep(0.005)
                continue
            try:
                #print("[debug] sending audio chunk...")
                pcm = self.audio.read_frames(CHUNK)
                self._send(b"AUD0", pcm)
                #print("[satellite] Sent AUD0 chunk")
            except Exception:
                break

    def on_barge_in(self, reason="wake"):
        """
        Interrupt current TTS and prepare to capture user speech immediately.
        """
        was_speaking = self.speaking_event.is_set()

        # Stop local output now
        self.playback.flush()
        self.speaking_event.clear()

        try:
            if self.sock and self._ready_event.is_set():
                if was_speaking:
                    print(f"[satellite] Barge-in ({reason}): interrupting TTS")
                    self._drop_tts_until_ready = True
                    self._send(b"INT0")
            else:
                print(f"[satellite] Barge-in ({reason}): sending WAKE")
                self._send(b"WAKE")
        except Exception:
            pass

    def _wake_vad_loop(self):
        """
        Runs continuously. While TTS is playing, listen for the wake word.
        """
        while True:
            if not self._wake_enabled.is_set():
                time.sleep(0.1)
                continue

            pcm = self.audio.read_frames(self.frame_len)
            frame = np.frombuffer(pcm, dtype=np.int16)

            # Wake word
            if self.porcupine.process(frame) >= 0:
                self.on_barge_in(reason="wake")
                continue

            # Optional: implement non-wake speech-over-TTS RMS barge-in here
            # using 'frame' with your level_threshold

    def wake_and_talk_once(self):
        """
        Blocks until wake word is detected, performs WAKE handshake,
        waits for RDY0, streams an utterance, and then returns.
        """
        print("[satellite] Listening for wake word...")
        while True:
            pcm = self.audio.read_frames(self.frame_len)
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

        # Wait for RDY0
        print("[satellite] Waiting for RDY0 from server...")
        if not self._ready_event.wait(timeout=5.0):
            print("[satellite] Timed out waiting for RDY0.")
            self.close()
            return

        print("[satellite] RDY0 received. Start talking.")
        # Stream until server ends the session (CLOS) or socket error
        self._stream_audio_session()

        # Wait for local playback to drain before closing
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if not self.speaking_event.is_set() and self.playback.queue.empty():
                break
            time.sleep(0.05)

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
    )
    client.run_forever()
