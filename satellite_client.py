import socket
import struct
import threading
import queue
import time
from typing import Tuple, Optional
from enum import Enum, auto

import numpy as np
import pyaudio
import yaml
import openwakeword

from pathlib import Path
import importlib.resources as ir

# -------------------------
# OpenWakeWord resource check
# -------------------------
def oww_resources_ok() -> bool:
    with ir.as_file(ir.files("openwakeword") / "resources" / "models") as models_dir:
        models_dir = Path(models_dir)
        required = ["melspectrogram.onnx"]
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
# Audio constants
# -------------------------
MIC_RATE        = 16000
SPK_RATE        = 16000
SAMPLE_WIDTH    = 2        # int16
INPUT_CHANNELS  = 1
OUTPUT_CHANNELS = 1
FORMAT          = pyaudio.paInt16
CHUNK           = 1024     # mic frames sent to server per packet
OUTPUT_CHUNK    = 4096     # playback buffer size

INPUT_GAIN_TARGET_RMS  = 0.2
INPUT_GAIN_MAX_DB      = 20.0
INPUT_GAIN_NOISE_FLOOR = 0.05

# -------------------------
# State machine states
# -------------------------
class State(Enum):
    IDLE      = auto()   # waiting for wake word
    WAITING   = auto()   # wake detected, waiting for server greeting + RDY0
    LISTENING = auto()   # mic open, streaming AUD0 to server
    THINKING  = auto()   # server processing, mic closed
    SPEAKING  = auto()   # server sending TTS, playing audio
    CLOSING   = auto()   # CLOS received, draining and tearing down

# -------------------------
# AudioIO
# -------------------------
class AudioIO:
    def __init__(self):
        self.p = pyaudio.PyAudio()

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
            print("[audio] Output started.")
        except Exception as e:
            print(f"[audio] FAILED to open output stream: {e}")
            raise

        self._in_stream = None

    def open_input(self, frames_per_buffer: int):
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
                start=False,
            )
            self._in_stream.start_stream()
        except Exception as e:
            print(f"[audio] FAILED to open input stream: {e}")
            raise

    def close_input(self):
        if self._in_stream:
            try:
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

    def read_input(self, n_frames: int) -> bytes:
        if not self._in_stream:
            time.sleep(0.01)
            return b"\x00" * (n_frames * SAMPLE_WIDTH)

        raw = self._in_stream.read(n_frames, exception_on_overflow=False)

        # Auto-gain
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms >= INPUT_GAIN_NOISE_FLOOR:
            max_gain = 10.0 ** (INPUT_GAIN_MAX_DB / 20.0)
            gain = min(INPUT_GAIN_TARGET_RMS / rms, max_gain)
            arr = np.clip(arr * gain, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    def flush_input(self, n_frames: int, count: int):
        """Read and discard frames to clear ALSA/Pulse buffer."""
        for _ in range(count):
            try:
                self.read_input(n_frames)
            except Exception:
                break

    def play_bytes(self, pcm_bytes: bytes):
        self.out.write(pcm_bytes)

    def beep(self, freq=800.0, duration=0.15, volume=0.4, rate=16000):
        n = int(rate * duration)
        t = np.arange(n, dtype=np.float32) / rate
        wave = np.sin(2 * np.pi * freq * t) * volume
        pcm = (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        self.play_bytes(pcm)

    def close(self):
        try:
            self.close_input()
            if self.out:
                self.out.close()
            self.p.terminate()
        except Exception:
            pass

# -------------------------
# PlaybackThread
# -------------------------
class PlaybackThread(threading.Thread):
    def __init__(self, audio: AudioIO):
        super().__init__(daemon=True)
        self.audio = audio
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=-1)
        self._stop = threading.Event()
        self._playing = False
        self._last_write_t = 0.0
        self._idle_grace = 0.3  # seconds after last write before declaring done

    @property
    def is_playing(self) -> bool:
        return self._playing

    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
        """Block until the queue is empty and playback has gone quiet."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self._playing and self.queue.empty():
                return True
            time.sleep(0.02)
        return False

    def run(self):
        while not self._stop.is_set():
            try:
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                # Check idle grace — if we were playing and queue has been
                # empty long enough, mark as done
                if self._playing and (time.time() - self._last_write_t) >= self._idle_grace:
                    self._playing = False
                continue

            if data is None:
                break

            self._playing = True
            self.audio.play_bytes(data)
            self._last_write_t = time.time()

        self._playing = False

    def enqueue(self, data: bytes):
        self.queue.put_nowait(data)

    def stop(self):
        self._stop.set()
        self.queue.put_nowait(None)

# -------------------------
# SatelliteClient
# -------------------------
class SatelliteClient:
    def __init__(self, server_host: str, server_port: int, wake_cfg: dict):
        self.server_host = server_host
        self.server_port = server_port
        self.wake_cfg    = wake_cfg

        # Audio
        self.audio    = AudioIO()
        self.playback = PlaybackThread(self.audio)
        self.playback.start()

        # Wake word
        model_path            = wake_cfg.get("model_path", "./soopanova.onnx")
        self.wake_threshold   = float(wake_cfg.get("threshold", 0.05))
        self.wake_cooldown_s  = float(wake_cfg.get("cooldown_s", 0.8))
        self.oww_hop          = 1280
        ensure_oww_resources()
        self.oww = openwakeword.Model(wakeword_model_paths=[model_path])

        # Networking
        self.sock: Optional[socket.socket] = None

        # State machine
        self._state      = State.IDLE
        self._state_lock = threading.Lock()
        self._event_q: "queue.Queue[Tuple[str, bytes]]" = queue.Queue()

        # Mic send thread control
        self._mic_active = threading.Event()  # set = mic thread should send audio

        # Startup beeps
        print("[satellite] Starting up...")
        self.audio.beep(800, 0.12, 0.3)
        time.sleep(0.08)
        self.audio.beep(400, 0.12, 0.3)

        # Post session silence (to avoid clipping end of TTS or server beeps on next session's wake word)
        self.post_session_silence_s = 0.4

    # -------------------------
    # State management
    # -------------------------
    def _set_state(self, new_state: State):
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            print(f"[state] {old.name} → {new_state.name}")
            self._on_state_enter(new_state)

    def _on_state_enter(self, state: State):
        """Side effects when entering a state — beeps, mic control, lights hook."""
        if state == State.IDLE:
            self._mic_active.clear()
            self.audio.close_input()

        elif state == State.WAITING:
            self._mic_active.clear()
            # High beep: wake word acknowledged, handing to server
            self.audio.beep(1000, 0.08, 0.3)

        elif state == State.LISTENING:
            # Lower beep: ready to listen
            self.audio.beep(700, 0.12, 0.3)
            self.audio.flush_input(CHUNK, 4)
            self._mic_active.set()

        elif state == State.THINKING:
            self._mic_active.clear()
            # Thinking beep handled by THNK frame so it's timed to transcription

        elif state == State.SPEAKING:
            self._mic_active.clear()

        elif state == State.CLOSING:
            self._mic_active.clear()
            # Wait for any remaining TTS to finish, then play closing beeps
            self.playback.wait_until_done(timeout_s=30.0)
            for _ in range(3):
                self.audio.beep(300, 0.20, 0.6)
                time.sleep(0.15)

    # -------------------------
    # Networking
    # -------------------------
    def _connect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = socket.create_connection((self.server_host, self.server_port))

    def _disconnect(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _send(self, tag: bytes, payload: bytes = b""):
        self.sock.sendall(pack_frame(tag, payload))

    # -------------------------
    # Receiver thread
    # -------------------------
    def _recv_loop(self):
        """
        Purely receives frames and puts them on the event queue.
        No side effects here — the state machine handles everything.
        """
        try:
            while True:
                tag, payload = read_frame(self.sock)
                self._event_q.put((tag.decode("ascii", errors="replace"), payload))
        except Exception as e:
            print(f"[recv] exiting: {e}")
            self._event_q.put(("_DISCONNECT", b""))

    # -------------------------
    # Mic sender thread
    # -------------------------
    def _mic_sender_loop(self):
        """
        Runs continuously while a session is active.
        Only sends AUD0 when _mic_active is set (i.e. state == LISTENING).
        Discards audio when not active, but keeps the mic stream open/ready to minimise latency on state switch to LISTENING.
        """
        try:
            self.audio.open_input(frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"[mic] failed to open input: {e}")
            return

        while self.sock is not None:
            try:
                pcm =self.audio.read_input(CHUNK)
            except Exception as e:
                print(f"[mic] read error: {e}")
                break

            if self.sock is None:
                break

            if not self._mic_active.is_set():
                # Not in LISTENING state — discard to keep buffer clear
                continue

            try:
                self._send(b"AUD0", pcm)
            except Exception as e:
                print(f"[mic] send error: {e}")
                break
        
        print("[mic] sender thread exiting")

    # -------------------------
    # State machine event loop
    # -------------------------
    def _run_session(self):
        """
        Drives the state machine for a single session (wake → close).
        Blocks until the session ends.
        """
        # Start receiver and mic sender threads
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()

        mic_thread = threading.Thread(target=self._mic_sender_loop, daemon=True)
        mic_thread.start()

        self._set_state(State.WAITING)

        try:
            self._send(b"WAKE")
        except Exception as e:
            print(f"[session] Error sending WAKE: {e}")
            return

        # Process events until session ends
        while True:
            try:
                tag, payload = self._event_q.get(timeout=60.0)
            except queue.Empty:
                print("[session] Timed out waiting for server event.")
                break

            if tag == "_DISCONNECT":
                print("[session] Socket disconnected.")
                break

            elif tag == "TTS0":
                duration_s = len(payload) / (SPK_RATE * SAMPLE_WIDTH)
                #print(f"[recv] TTS0 {len(payload)} bytes = {duration_s:.2f}s")
                self.playback.enqueue(payload)
                if self._state != State.SPEAKING:
                    self._set_state(State.SPEAKING)

            elif tag == "BEEP":
                print(f"[recv] BEEP {len(payload)} bytes")
                self.playback.enqueue(payload)
                if self._state != State.SPEAKING:
                    self._set_state(State.SPEAKING)

            elif tag == "RDY0":
                print("[recv] RDY0 — waiting for playback to drain...")
                # Block here until all queued audio has played out
                self.playback.wait_until_done(timeout_s=30.0)
                self._set_state(State.LISTENING)

            elif tag == "THNK":
                print("[recv] THNK")
                self.audio.beep(1000, 0.12, 0.3)
                self._set_state(State.THINKING)

            elif tag == "CLOS":
                print("[recv] CLOS")
                self._set_state(State.CLOSING)
                break

            else:
                print(f"[recv] Unknown tag {tag!r}")

    # -------------------------
    # Wake word loop
    # -------------------------
    def _flush_wake_model(self, seconds: float = 2.0):
        for name in ("reset", "reset_states", "reset_state"):
            fn = getattr(self.oww, name, None)
            if callable(fn):
                fn()
                return
        n_samples = int(MIC_RATE * seconds)
        step      = self.oww_hop
        silence   = np.zeros(step, dtype=np.int16)
        for _ in range((n_samples + step - 1) // step):
            self.oww.predict(silence)

    def _wait_for_wake_word(self):
        """Opens mic, blocks until wake word detected, closes mic."""
        self.audio.open_input(frames_per_buffer=self.oww_hop)
        self.audio.flush_input(self.oww_hop, 3)

        print("[satellite] Listening for wake word...")
        next_allowed_t = 0.0

        while True:
            pcm   = self.audio.read_input(self.oww_hop)
            frame = np.frombuffer(pcm, dtype=np.int16)
            pred  = self.oww.predict(frame)

            if not pred:
                continue

            model_name = max(pred, key=pred.get)
            score      = float(pred[model_name])
            now        = time.time()

            if now < next_allowed_t:
                continue

            if score >= self.wake_threshold:
                print(f"[satellite] Wake word '{model_name}' detected (score={score:.3f})")
                next_allowed_t = now + self.wake_cooldown_s
                break

        # Close wake-word mic stream — session will reopen for CHUNK-sized reads
        self.audio.close_input()

    # -------------------------
    # Main loop
    # -------------------------
    def run_forever(self):
        try:
            while True:
                # Clear any stale events from previous session
                while not self._event_q.empty():
                    try:
                        self._event_q.get_nowait()
                    except Exception:
                        break

                self._set_state(State.IDLE)
                self._mic_active.clear()

                self._wait_for_wake_word()
                self._connect()

                # Track threads so we can wait for them to die
                recv_thread = None
                mic_thread  = None

                try:
                    self._run_session()
                except Exception as e:
                    print(f"[satellite] Session error: {e}")
                finally:
                    # Ensure mic is stopped and socket closed before anything else
                    self._mic_active.clear()
                    self._disconnect()

                    # Give threads a moment to exit cleanly
                    time.sleep(0.3)

                    # Reopen the input stream fresh for wake word listening
                    try:
                        self.audio.open_input(frames_per_buffer=self.oww_hop)
                    except Exception as e:
                        print(f"[satellite] Failed to reopen mic: {e}")

                self._flush_wake_model(seconds=2.0)
                time.sleep(self.post_session_silence_s if hasattr(self, 'post_session_silence_s') else 0.4)
                print("[satellite] Session complete, back to wake word listening.")

        except KeyboardInterrupt:
            print("\n[satellite] Stopping...")
        finally:
            self.playback.stop()
            self.audio.close()

# -------------------------
# Config + entrypoint
# -------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config()
    client = SatelliteClient(
        server_host=cfg["voice_server"]["host"],
        server_port=int(cfg["voice_server"]["port"]),
        wake_cfg=cfg.get("wake_word", {}),
    )
    client.run_forever()