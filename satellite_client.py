"""
satellite_client.py — Edge speaker endpoint (runs on the satellite device).

Persistent-connection model
─────────────────────────────
The satellite connects to the server ONCE at startup and keeps that connection
alive for the lifetime of the process. It does NOT connect/disconnect per
wake-word session. Instead:

  1. On startup: connect → send HELO with endpoint_id → enter IDLE.
  2. On wake word (local): send WAKE on the existing connection → enter WAITING.
  3. Session runs normally (LISTENING → THINKING → SPEAKING → back to IDLE).
  4. On CLOS from server: session ends but connection stays open → back to IDLE.
  5. On disconnect: reconnect loop with backoff → re-register with HELO.
  6. Server can initiate a session by sending CALL to the satellite at any time
     while in IDLE. The satellite responds as if it had detected a wake word.

This means the server always knows which endpoints are online (registry of
connected ClientSession objects keyed by endpoint_id) and can initiate
conversations, announcements, or alerts without waiting for the user to speak.

New frames added to the protocol:

  Client → Server:
    HELO     UTF-8 payload = endpoint_id (sent once on connect / reconnect)
    WAKE     no payload — wake word detected, start a session
    AUD0     int16 mono 16kHz PCM chunk
    INT0     no payload — barge-in
    STOP     no payload — force end-of-utterance flush

  Server → Client:
    RDY0     no payload — ready for audio input
    TTS0     int16 mono 16kHz PCM — synthesised speech
    BEEP     int16 mono 16kHz PCM — UX sound effect
    THNK     no payload — server is thinking
    CLOS     no payload — session ended (connection stays open)
    CALL     UTF-8 payload = optional announcement text, or empty for plain wake

Threading model (unchanged from before):
  - Main thread: wake word loop + session state machine.
  - _recv_loop thread: blocking recv → _event_q.
  - _mic_sender_loop thread: mic → AUD0. Controlled by _mic_active.
  - PlaybackThread: PCM queue → speaker.

The key difference from before: _connect() and _disconnect() are only called
from the reconnect loop in run_forever(), not from the session teardown path.
CLOS handling no longer closes the socket — it just resets session state and
returns to IDLE/wake-word detection while the connection stays alive.
"""

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


# ── OpenWakeWord resource check ───────────────────────────────────────────────

def oww_resources_ok() -> bool:
    with ir.as_file(ir.files("openwakeword") / "resources" / "models") as models_dir:
        models_dir = Path(models_dir)
        return all((models_dir / f).is_file() for f in ["melspectrogram.onnx"])

def ensure_oww_resources():
    if oww_resources_ok():
        return
    import openwakeword.utils
    openwakeword.utils.download_models()


# ── Frame protocol ────────────────────────────────────────────────────────────

FRAME_HDR = struct.Struct("<4sI")   # 4-byte tag + uint32 payload length

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
    header     = read_exactly(sock, FRAME_HDR.size)
    tag, length = FRAME_HDR.unpack(header)
    payload    = read_exactly(sock, length) if length else b""
    return tag, payload


# ── Audio constants ───────────────────────────────────────────────────────────

MIC_RATE        = 16000
SPK_RATE        = 16000
SAMPLE_WIDTH    = 2
INPUT_CHANNELS  = 1
OUTPUT_CHANNELS = 1
FORMAT          = pyaudio.paInt16
CHUNK           = 1024
OUTPUT_CHUNK    = 4096

INPUT_GAIN_TARGET_RMS        = 0.2
INPUT_GAIN_MAX_DB            = 12.0
INPUT_GAIN_NOISE_FLOOR       = 0.03
INPUT_GAIN_SMOOTHING_ATTACK  = 0.70
INPUT_GAIN_SMOOTHING_RELEASE = 0.95

DEBUG_GAIN = False
DEBUG_CLIP = False

def _gain_bar(gain: float, rms: float = 0.0, max_db: float = INPUT_GAIN_MAX_DB, width: int = 40) -> str:
    max_gain = 10.0 ** (max_db / 20.0)
    ratio    = min(gain / max_gain, 1.0)
    filled   = int(ratio * width)
    return (f"\r[{'#' * filled}{' ' * (width - filled)}] "
            f"{gain:.3f}x ({20 * np.log10(max(gain, 1e-9)):.1f}dB) rms={rms:.4f}")


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    """
    States of the satellite.

      IDLE      — connected to server, listening for wake word or CALL frame.
      WAITING   — WAKE sent, waiting for server greeting + RDY0.
      LISTENING — mic open, streaming AUD0 to server.
      THINKING  — transcript sent, waiting for TTS response.
      SPEAKING  — receiving and playing TTS0/BEEP frames.
      CLOSING   — CLOS received, draining audio before returning to IDLE.
                  (Connection stays open — this is NOT a disconnect state.)
    """
    IDLE      = auto()
    WAITING   = auto()
    LISTENING = auto()
    THINKING  = auto()
    SPEAKING  = auto()
    CLOSING   = auto()


# ── AudioIO ───────────────────────────────────────────────────────────────────

class AudioIO:
    """Thin PyAudio wrapper. Output stream stays open for the life of the process."""

    def __init__(self):
        self.p             = pyaudio.PyAudio()
        self._current_gain = 1.0

        self.out = self.p.open(
            format=pyaudio.paInt16, channels=OUTPUT_CHANNELS, rate=SPK_RATE,
            output=True, frames_per_buffer=OUTPUT_CHUNK, start=False,
        )
        self.out.start_stream()
        print("[audio] Output started.")
        self._in_stream = None

    def open_input(self, frames_per_buffer: int):
        if self._in_stream:
            try: self._in_stream.close()
            except Exception: pass
            self._in_stream = None
        self._in_stream = self.p.open(
            format=pyaudio.paInt16, channels=1, rate=MIC_RATE,
            input=True, frames_per_buffer=frames_per_buffer, start=False,
        )
        self._in_stream.start_stream()

    def close_input(self):
        if self._in_stream:
            try: self._in_stream.close()
            except Exception: pass
            self._in_stream = None

    def read_input_raw(self, n_frames: int) -> bytes:
        """Raw read, no gain — for ASR audio sent to Whisper."""
        if not self._in_stream:
            time.sleep(0.01)
            return b"\x00" * (n_frames * SAMPLE_WIDTH)
        return self._in_stream.read(n_frames, exception_on_overflow=False)

    def read_input(self, n_frames: int) -> bytes:
        """Auto-gain read — for wake word detection (OWW needs boosted signal)."""
        if not self._in_stream:
            time.sleep(0.01)
            return b"\x00" * (n_frames * SAMPLE_WIDTH)
        raw = self._in_stream.read(n_frames, exception_on_overflow=False)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if rms >= INPUT_GAIN_NOISE_FLOOR:
            max_gain    = 10.0 ** (INPUT_GAIN_MAX_DB / 20.0)
            target_gain = min(INPUT_GAIN_TARGET_RMS / rms, max_gain)
        else:
            target_gain = self._current_gain
        alpha = INPUT_GAIN_SMOOTHING_ATTACK if target_gain < self._current_gain else INPUT_GAIN_SMOOTHING_RELEASE
        self._current_gain = alpha * self._current_gain + (1.0 - alpha) * target_gain
        arr = np.clip(arr * self._current_gain, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()

    def flush_input(self, n_frames: int, count: int):
        for _ in range(count):
            try: self.read_input(n_frames)
            except Exception: break

    def input_available_frames(self) -> int:
        return self._in_stream.get_read_available() if self._in_stream else 0

    def play_bytes(self, pcm_bytes: bytes):
        self.out.write(pcm_bytes)

    def beep(self, freq=800.0, duration=0.15, volume=0.4, rate=16000):
        """Synchronous beep — played immediately, bypasses PlaybackThread queue."""
        n    = int(rate * duration)
        t    = np.arange(n, dtype=np.float32) / rate
        wave = np.sin(2 * np.pi * freq * t) * volume
        self.play_bytes((np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    def beep_pcm(self, freq=800.0, duration=0.15, volume=0.4, rate=16000) -> bytes:
        """Return beep as PCM bytes for enqueuing in PlaybackThread."""
        n    = int(rate * duration)
        t    = np.arange(n, dtype=np.float32) / rate
        wave = np.sin(2 * np.pi * freq * t) * volume
        return (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def close(self):
        try:
            self.close_input()
            if self.out: self.out.close()
            self.p.terminate()
        except Exception: pass


# ── PlaybackThread ────────────────────────────────────────────────────────────

class PlaybackThread(threading.Thread):
    """Drains a PCM byte queue and writes to the audio output."""

    def __init__(self, audio: AudioIO):
        super().__init__(daemon=True)
        self.audio         = audio
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._stop         = threading.Event()
        self._playing      = False
        self._last_write_t = 0.0
        self._idle_grace   = 0.3

    @property
    def is_playing(self) -> bool:
        return self._playing

    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
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
                if self._playing and (time.time() - self._last_write_t) >= self._idle_grace:
                    self._playing = False
                continue
            if data is None:
                break
            self._playing      = True
            self.audio.play_bytes(data)
            self._last_write_t = time.time()
        self._playing = False

    def enqueue(self, data: bytes):
        self.queue.put_nowait(data)

    def stop(self):
        self._stop.set()
        self.queue.put_nowait(None)


# ── SatelliteClient ───────────────────────────────────────────────────────────

class SatelliteClient:
    """
    Persistent-connection satellite client.

    Lifecycle:
      startup → connect → HELO → IDLE (wake word loop)
                                   ↓ wake word detected (or CALL received)
                               WAKE sent → session → CLOS received
                                   ↓
                               back to IDLE (same connection)

      On disconnect at any point: reconnect loop with exponential backoff,
      then re-register with HELO and return to IDLE.
    """

    def __init__(self, server_host: str, server_port: int,
                 wake_cfg: dict, endpoint_id: str):
        self.server_host = server_host
        self.server_port = server_port
        self.wake_cfg    = wake_cfg

        # endpoint_id uniquely identifies this satellite to the server.
        # Sent in the HELO frame on connect. Should be stable across restarts
        # (e.g. hostname, MAC address, or a configured name like "kitchen").
        self.endpoint_id = endpoint_id

        # ── Audio ─────────────────────────────────────────────────────────────
        self.audio    = AudioIO()
        self.playback = PlaybackThread(self.audio)
        self.playback.start()

        # ── Wake word ─────────────────────────────────────────────────────────
        self.wake_threshold  = float(wake_cfg.get("threshold", 0.05))
        self.wake_cooldown_s = float(wake_cfg.get("cooldown_s", 0.8))
        self.oww_hop         = 2560
        ensure_oww_resources()
        self.oww = None

        # ── Networking ────────────────────────────────────────────────────────
        self.sock: Optional[socket.socket] = None

        # ── State machine ─────────────────────────────────────────────────────
        self._state      = State.IDLE
        self._state_lock = threading.Lock()

        # _event_q carries (tag_str, payload_bytes) from _recv_loop to the
        # session state machine. Also carries synthetic events:
        #   "_DISCONNECT" — socket closed
        #   "_WAKE"       — local wake word detected (posted by wake word loop)
        self._event_q: "queue.Queue[Tuple[str, bytes]]" = queue.Queue()

        # ── Mic send thread control ───────────────────────────────────────────
        self._mic_active = threading.Event()   # set = send AUD0 frames
        self._mic_stop   = threading.Event()   # set = mic thread should exit

        # ── Startup beeps ─────────────────────────────────────────────────────
        print(f"[satellite] Starting up (endpoint_id={endpoint_id!r})...")
        self.audio.beep(800, 0.12, 0.3)
        time.sleep(0.08)
        self.audio.beep(400, 0.12, 0.3)

    # ── State management ──────────────────────────────────────────────────────

    def _set_state(self, new_state: State):
        with self._state_lock:
            old         = self._state
            self._state = new_state
        if old != new_state:
            print(f"[state] {old.name} → {new_state.name}")
            self._on_state_enter(new_state)

    def _on_state_enter(self, state: State):
        """Side-effects on state entry: beeps, mic control, future LED hooks."""
        if state == State.IDLE:
            # Connection is alive; we're just waiting for wake word or CALL.
            self._mic_active.clear()

        elif state == State.WAITING:
            # WAKE sent — waiting for server greeting.
            self._mic_active.clear()
            self.audio.beep(1000, 0.08, 0.3)

        elif state == State.LISTENING:
            # RDY0 received, playback drained — open the mic.
            self.audio.beep(700, 0.12, 0.3)
            self._mic_active.set()

        elif state == State.THINKING:
            self._mic_active.clear()

        elif state == State.SPEAKING:
            self._mic_active.clear()

        elif state == State.CLOSING:
            # CLOS received — drain audio and play closing beeps, then go IDLE.
            # The connection is NOT closed here.
            self._mic_active.clear()
            self.playback.wait_until_done(timeout_s=30.0)
            for _ in range(3):
                self.audio.beep(300, 0.20, 0.6)
                time.sleep(0.15)

    # ── Networking ────────────────────────────────────────────────────────────

    def _connect(self):
        """Open (or reopen) the TCP connection and register with HELO."""
        if self.sock:
            try: self.sock.close()
            except Exception: pass
        self.sock = socket.create_connection((self.server_host, self.server_port))
        # Register immediately so the server knows who we are.
        self._send(b"HELO", self.endpoint_id.encode("utf-8"))
        print(f"[satellite] Connected and registered as {self.endpoint_id!r}")

    def _disconnect(self):
        try:
            if self.sock: self.sock.close()
        except Exception: pass
        self.sock = None

    def _send(self, tag: bytes, payload: bytes = b""):
        self.sock.sendall(pack_frame(tag, payload))

    # ── Receiver thread ───────────────────────────────────────────────────────

    def _recv_loop(self):
        """
        Blocking receiver: reads frames and posts them to _event_q.
        Runs for the lifetime of the connection (not just during a session).
        This means CALL frames received while IDLE are queued and processed
        by the main loop, which wakes from its wake-word wait.
        """
        try:
            while True:
                tag, payload = read_frame(self.sock)
                self._event_q.put((tag.decode("ascii", errors="replace"), payload))
        except Exception as e:
            print(f"[recv] exiting: {e}")
            self._event_q.put(("_DISCONNECT", b""))

    # ── Mic sender thread ─────────────────────────────────────────────────────

    def _mic_sender_loop(self):
        """
        Reads mic and sends AUD0 frames while _mic_active is set.
        Runs for the lifetime of the connection (started once per connection,
        not once per session). _mic_stop terminates it on disconnect.
        """
        self._mic_stop.clear()
        try:
            self.audio.open_input(frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"[mic] failed to open input: {e}")
            return

        self.audio.flush_input(CHUNK, 4)

        while not self._mic_stop.is_set() and self.sock is not None:
            if not self._mic_active.is_set():
                self._mic_active.wait(timeout=0.1)
                if self._mic_active.is_set():
                    # Reopen the stream — the wake word listener may have
                    # closed and replaced it while we were inactive.
                    self.audio.open_input(frames_per_buffer=CHUNK)
                    self.audio.flush_input(CHUNK, 4)
                continue
            try:
                pcm = self.audio.read_input_raw(CHUNK)
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(arr ** 2)))
                
                # debugging audio levels
                #if rms > 0.01:
                #    print(f"[mic] rms={rms:.4f}")
            except Exception as e:
                print(f"[mic] read error: {e}")
                break
            if self.sock is None or self._mic_stop.is_set():
                break
            if not self._mic_active.is_set():
                continue
            try:
                self._send(b"AUD0", pcm)
            except Exception as e:
                print(f"[mic] send error: {e}")
                break

        self.audio.close_input()
        print("[mic] sender thread exiting")

    # ── Wake word detection ───────────────────────────────────────────────────

    def _start_wake_word_listener(self):
        """
        Start a background thread that listens for the wake word and posts a
        synthetic "_WAKE" event to _event_q when detected.

        This runs concurrently with the main loop's event.get() so that CALL
        frames from the server can also wake the satellite — whichever arrives
        first wins.

        The thread exits as soon as it detects a wake word (or the oww_stop
        event is set), so it doesn't run during sessions.
        """
        self._oww_stop = threading.Event()

        def _wake_thread():
            model_path = self.wake_cfg.get("model_path", "./soopanova.onnx")
            oww = openwakeword.Model(
                wakeword_model_paths=[model_path],
                vad_threshold=float(self.wake_cfg.get("vad_threshold", 0.5)),
            )
            try:
                self.audio.open_input(frames_per_buffer=self.oww_hop)
                self.audio.flush_input(self.oww_hop, 3)
                next_allowed_t = 0.0

                while not self._oww_stop.is_set():
                    pcm  = self.audio.read_input(self.oww_hop)
                    available = self.audio.input_available_frames()
                    if available > self.oww_hop * 2:
                        self.audio.flush_input(self.oww_hop, available // self.oww_hop)
                        continue

                    frame = np.frombuffer(pcm, dtype=np.int16)
                    pred  = oww.predict(frame)
                    if not pred:
                        continue

                    model_name = max(pred, key=pred.get)
                    score      = float(pred[model_name])
                    now        = time.time()

                    if now < next_allowed_t:
                        continue

                    if score >= self.wake_threshold:
                        print(f"\n[satellite] Wake word detected (score={score:.3f})")
                        next_allowed_t = now + self.wake_cooldown_s
                        self._event_q.put(("_WAKE", b""))
                        break   # thread exits; main loop handles the rest
                    elif score > 0.002:
                        print(f"\n[satellite] Poor recognition (score={score:.3f})")
            finally:
                self.audio.close_input()
                oww = None

        self._wake_thread = threading.Thread(target=_wake_thread, daemon=True)
        self._wake_thread.start()

    def _stop_wake_word_listener(self):
        """Stop the wake word thread and wait for it to release the mic."""
        if hasattr(self, '_oww_stop'):
            self._oww_stop.set()
        if hasattr(self, '_wake_thread') and self._wake_thread.is_alive():
            print("[satellite] waiting for wake thread to release mic...")
            self._wake_thread.join(timeout=2.0)

    # ── Connection-level loop ─────────────────────────────────────────────────

    def _run_connection(self):
        """
        Run the full lifecycle for one persistent connection.

        Starts _recv_loop and _mic_sender_loop once. Then alternates between:
          - IDLE phase: wake word listener + waiting for "_WAKE" or "CALL" event.
          - Session phase: handles WAITING → LISTENING → THINKING → SPEAKING → CLOSING.

        Returns when the socket disconnects (_DISCONNECT event).
        """
        # Start persistent threads for this connection.
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()

        mic_thread = threading.Thread(target=self._mic_sender_loop, daemon=True)
        mic_thread.start()

        self._set_state(State.IDLE)

        while True:
            # ── IDLE phase: wait for wake word or server-initiated CALL ───────
            print(f"[satellite] IDLE — listening for wake word or CALL from server...")
            self._start_wake_word_listener()

            # Block until something wakes us: local wake word, server CALL,
            # or a disconnect.
            channel_already_open = False
            while True:
                try:
                    tag, payload = self._event_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if tag == "_DISCONNECT":
                    self._stop_wake_word_listener()
                    return   # bubble up to reconnect loop

                elif tag in ("_WAKE", "CALL"):
                    # Wake word detected locally, OR server pushed a CALL frame.
                    self._stop_wake_word_listener()
                    if tag == "CALL" and payload:
                        # Server can include optional text in CALL payload,
                        # e.g. an announcement. Print it for now; future work
                        # could pre-populate it as the first assistant turn.
                        print(f"[satellite] Server-initiated CALL: {payload.decode('utf-8', errors='replace')!r}")
                    break

                elif tag == "TTS0" or tag == "BEEP":
                    # Server greeting ("I'm here") arriving before session loop.
                    # Play it through the normal playback queue.
                    self.playback.enqueue(payload)

                elif tag == "RDY0":
                    # Server is ready — wait for greeting to finish playing,
                    # then break as if a wake just happened so we enter the
                    # session loop immediately.
                    self.playback.wait_until_done(timeout_s=10.0)
                    self._stop_wake_word_listener()
                    channel_already_open = True
                    break

                else:
                    # Any other frame arriving while IDLE is unexpected but
                    # harmless — log and discard.
                    print(f"[satellite] Unexpected frame while IDLE: {tag!r}")

            # ── Session phase: send WAKE and drive state machine ──────────────
            self._set_state(State.WAITING)
            if channel_already_open:
                # Server already greeted us and sent RDY0 — go straight to
                # LISTENING without sending WAKE again.
                self.playback.wait_until_done(timeout_s=10.0)
                self._set_state(State.LISTENING)
            else:
                try:
                    self._send(b"WAKE")
                except Exception as e:
                    print(f"[session] Error sending WAKE: {e}")
                    return

            session_done = False
            while not session_done:
                try:
                    tag, payload = self._event_q.get(timeout=60.0)
                except queue.Empty:
                    print("[session] Timed out waiting for server event.")
                    return

                if tag == "_DISCONNECT":
                    return

                elif tag == "TTS0":
                    self.playback.enqueue(payload)
                    if self._state != State.SPEAKING:
                        self._set_state(State.SPEAKING)

                elif tag == "BEEP":
                    self.playback.enqueue(payload)
                    if self._state != State.SPEAKING:
                        self._set_state(State.SPEAKING)

                elif tag == "RDY0":
                    # Wait for playback to drain before opening the mic.
                    print("[recv] RDY0 — waiting for playback to drain...")
                    self.playback.wait_until_done(timeout_s=30.0)
                    self._set_state(State.LISTENING)

                elif tag == "THNK":
                    beep_pcm = self.audio.beep_pcm(1000, 0.12, 0.3)
                    self.playback.enqueue(beep_pcm)
                    self._set_state(State.THINKING)

                elif tag == "CLOS":
                    # Session ended. Connection stays open.
                    # _on_state_enter(CLOSING) drains audio and plays closing beeps.
                    print("[recv] CLOS — session ended, connection persists")
                    self._set_state(State.CLOSING)
                    session_done = True   # break inner loop, return to IDLE phase

                else:
                    print(f"[recv] Unknown tag {tag!r}")

            # Session done — loop back to IDLE (wake word listening).
            self._set_state(State.IDLE)

    # ── Main loop with reconnect ──────────────────────────────────────────────

    def run_forever(self):
        """
        Top-level loop: connect → run → reconnect on failure.

        Exponential backoff on reconnect: 2s, 4s, 8s … up to 60s.
        On successful reconnect, HELO is sent inside _connect() so the server
        re-registers this endpoint immediately.
        """
        backoff = 2.0
        try:
            while True:
                # Drain stale events from any previous connection.
                while not self._event_q.empty():
                    self._event_q.get_nowait()

                try:
                    print(f"[satellite] Connecting to {self.server_host}:{self.server_port}...")
                    self._connect()
                    backoff = 2.0   # reset backoff on successful connect
                except Exception as e:
                    print(f"[satellite] Connection failed: {e}. Retrying in {backoff:.0f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue

                try:
                    self._run_connection()
                finally:
                    self._mic_active.clear()
                    self._mic_stop.set()
                    self._disconnect()
                    time.sleep(1)   # give mic thread time to exit

                print(f"[satellite] Disconnected. Reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

        except KeyboardInterrupt:
            print("\n[satellite] Stopping...")
        finally:
            self.playback.stop()
            self.audio.close()


# ── Config + entrypoint ───────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config()

    gain_cfg = cfg.get("input_gain", {})
    INPUT_GAIN_TARGET_RMS        = gain_cfg.get("target_rms",         INPUT_GAIN_TARGET_RMS)
    INPUT_GAIN_MAX_DB            = gain_cfg.get("max_db",             INPUT_GAIN_MAX_DB)
    INPUT_GAIN_NOISE_FLOOR       = gain_cfg.get("noise_floor",        INPUT_GAIN_NOISE_FLOOR)
    INPUT_GAIN_SMOOTHING_ATTACK  = gain_cfg.get("smoothing_attack",   INPUT_GAIN_SMOOTHING_ATTACK)
    INPUT_GAIN_SMOOTHING_RELEASE = gain_cfg.get("smoothing_release",  INPUT_GAIN_SMOOTHING_RELEASE)

    DEBUG_GAIN = cfg.get("debug", {}).get("gain_bar",      False)
    DEBUG_CLIP = cfg.get("debug", {}).get("clip_warnings", False)

    # endpoint_id: stable identifier for this satellite. Falls back to hostname.
    import socket as _socket
    endpoint_id = cfg.get("endpoint_id", _socket.gethostname())

    client = SatelliteClient(
        server_host=cfg["voice_server"]["host"],
        server_port=int(cfg["voice_server"]["port"]),
        wake_cfg=cfg.get("wake_word", {}),
        endpoint_id=endpoint_id,
    )
    client.run_forever()