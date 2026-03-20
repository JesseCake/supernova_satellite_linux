#!/usr/bin/env python3
"""
wake_test.py — Wake word detection tester with auto-gain and beep feedback.

Usage:
    python wake_test.py [--config config.yaml] [--gain-bar]

Reads the same config.yaml as the main satellite client.
"""

import argparse
import struct
import time
from pathlib import Path
import importlib.resources as ir

import numpy as np
import pyaudio
import openwakeword

# -------------------------
# Audio constants
# -------------------------
MIC_RATE     = 16000
SAMPLE_WIDTH = 2        # int16
FORMAT       = pyaudio.paInt16
OWW_HOP      = 1820     # frames per prediction hop
OUTPUT_CHUNK = 4096

# -------------------------
# Auto-gain defaults
# -------------------------
INPUT_GAIN_TARGET_RMS        = 0.2
INPUT_GAIN_MAX_DB            = 12.0
INPUT_GAIN_NOISE_FLOOR       = 0.03
INPUT_GAIN_SMOOTHING_ATTACK  = 0.70
INPUT_GAIN_SMOOTHING_RELEASE = 0.95


# -------------------------
# OpenWakeWord resource check
# -------------------------
def ensure_oww_resources():
    with ir.as_file(ir.files("openwakeword") / "resources" / "models") as models_dir:
        required = ["melspectrogram.onnx"]
        if not all((Path(models_dir) / f).is_file() for f in required):
            import openwakeword.utils
            openwakeword.utils.download_models()


# -------------------------
# Beep helpers
# -------------------------
def make_beep_pcm(freq=800.0, duration=0.15, volume=0.4, rate=16000) -> bytes:
    n = int(rate * duration)
    t = np.arange(n, dtype=np.float32) / rate
    wave = np.sin(2 * np.pi * freq * t) * volume
    return (np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def play_beep(out_stream: pyaudio.Stream, freq=800.0, duration=0.15, volume=0.4):
    out_stream.write(make_beep_pcm(freq, duration, volume))


# -------------------------
# Auto-gain processor
# -------------------------
class AutoGain:
    def __init__(self, target_rms, max_db, noise_floor, attack, release):
        self.target_rms  = target_rms
        self.max_gain    = 10.0 ** (max_db / 20.0)
        self.noise_floor = noise_floor
        self.attack      = attack
        self.release     = release
        self._gain       = 1.0

    def process(self, raw: bytes) -> tuple[bytes, float, float]:
        """Returns (gained_pcm, current_gain, rms)."""
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2)))

        if rms >= self.noise_floor:
            target = min(self.target_rms / rms, self.max_gain)
        else:
            target = self._gain  # hold during silence

        alpha = self.attack if target < self._gain else self.release
        self._gain = alpha * self._gain + (1.0 - alpha) * target

        arr = np.clip(arr * self._gain, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16).tobytes()
        return pcm, self._gain, rms

    def gain_bar(self, width=36) -> str:
        ratio  = min(self._gain / self.max_gain, 1.0)
        filled = int(ratio * width)
        db     = 20 * np.log10(max(self._gain, 1e-9))
        return f"[{'█' * filled}{'░' * (width - filled)}] {self._gain:.3f}x ({db:+.1f}dB)"


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Wake word detection tester")
    parser.add_argument("--config",   default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--gain-bar", action="store_true",   help="Show live gain bar")
    args = parser.parse_args()

    import yaml
    if not Path(args.config).is_file():
        parser.error(f"Config file not found: {args.config!r}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    cfg_wake = cfg.get("wake_word", {})
    cfg_gain = cfg.get("input_gain", {})

    model_path = cfg_wake.get("model_path", "./soopanova.onnx")
    threshold  = float(cfg_wake.get("threshold", 0.05))
    vad_thresh = float(cfg_wake.get("vad_threshold", 0.5))
    cooldown_s = float(cfg_wake.get("cooldown_s", 0.8))
    show_bar   = args.gain_bar or cfg.get("debug", {}).get("gain_bar", False)

    gain = AutoGain(
        target_rms = cfg_gain.get("target_rms",         INPUT_GAIN_TARGET_RMS),
        max_db     = cfg_gain.get("max_db",             INPUT_GAIN_MAX_DB),
        noise_floor= cfg_gain.get("noise_floor",        INPUT_GAIN_NOISE_FLOOR),
        attack     = cfg_gain.get("smoothing_attack",   INPUT_GAIN_SMOOTHING_ATTACK),
        release    = cfg_gain.get("smoothing_release",  INPUT_GAIN_SMOOTHING_RELEASE),
    )

    print(f"[wake_test] Model     : {model_path}")
    print(f"[wake_test] Threshold : {threshold}")
    print(f"[wake_test] VAD thresh: {vad_thresh}")
    print(f"[wake_test] Cooldown  : {cooldown_s}s")
    print(f"[wake_test] Gain bar  : {show_bar}")

    ensure_oww_resources()

    pa = pyaudio.PyAudio()
    out = pa.open(format=FORMAT, channels=1, rate=MIC_RATE,
                  output=True, frames_per_buffer=OUTPUT_CHUNK)
    out.start_stream()

    # Startup beep
    play_beep(out, freq=800, duration=0.12, volume=0.3)

    try:
        while True:
            print("\n[wake_test] Initialising wake word model...", end=" ", flush=True)
            oww = openwakeword.Model(
                wakeword_model_paths=[model_path],
                vad_threshold=vad_thresh,
            )
            print("ready.")
            print("[wake_test] Listening for wake word  (Ctrl-C to quit)\n")

            mic = pa.open(format=FORMAT, channels=1, rate=MIC_RATE,
                          input=True, frames_per_buffer=OWW_HOP)
            mic.start_stream()

            # Flush initial buffer
            for _ in range(3):
                mic.read(OWW_HOP, exception_on_overflow=False)

            next_allowed = 0.0
            detected     = False

            try:
                while not detected:
                    raw = mic.read(OWW_HOP, exception_on_overflow=False)

                    # Drain lag
                    available = mic.get_read_available()
                    if available > OWW_HOP * 2:
                        for _ in range(available // OWW_HOP):
                            mic.read(OWW_HOP, exception_on_overflow=False)
                        continue

                    pcm, cur_gain, rms = gain.process(raw)

                    if show_bar:
                        print(f"\r{gain.gain_bar()} rms={rms:.4f}", end="", flush=True)

                    frame = np.frombuffer(pcm, dtype=np.int16)
                    pred  = oww.predict(frame)
                    if not pred:
                        continue

                    now        = time.time()
                    model_name = max(pred, key=pred.get)
                    score      = float(pred[model_name])

                    if now < next_allowed:
                        continue

                    if score >= threshold:
                        if show_bar:
                            print()
                        print(f"  ✓  DETECTED  '{model_name}'  score={score:.4f}")
                        play_beep(out, freq=1000, duration=0.12, volume=0.4)
                        time.sleep(0.05)
                        play_beep(out, freq=1200, duration=0.08, volume=0.3)
                        next_allowed = now + cooldown_s
                        # Don't break — keep listening so you can test multiple times

                    elif score > 0.001:
                        if show_bar:
                            print()
                        print(f"  ~  partial   '{model_name}'  score={score:.4f}")

            finally:
                mic.close()

            # Brief pause before re-initialising model (shouldn't normally reach here)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[wake_test] Stopped.")
    finally:
        play_beep(out, freq=400, duration=0.15, volume=0.3)
        out.close()
        pa.terminate()


if __name__ == "__main__":
    main()