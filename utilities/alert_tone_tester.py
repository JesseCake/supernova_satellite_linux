#!/usr/bin/env python3
"""
test_alert_tone.py — Standalone test for the XDR/SDR cassette inbound alert tone.

Run directly to hear the tone:
    python3 test_alert_tone.py

Optional arguments:
    --repeats N     How many times to play the sequence (default: 2)
    --volume N      Volume 0.0–1.0 (default: 0.85)
    --rate N        Sample rate in Hz (default: 16000)
    --no-ramp       Disable high-frequency volume ramp
"""

import argparse
import time
import numpy as np
import pyaudio

OUTPUT_CHUNK = 4096


def play_alert(repeats: int = 2, volume: float = 0.45, rate: int = 16000, ramp: bool = True):
    """
    XDR/SDR cassette tone burst — the ascending quality-control tones
    used by Capitol/EMI on pre-recorded cassettes. 15 sine wave tones
    stepping up from 50Hz to 18300Hz, each 127ms with 23ms gaps.
    Runs `repeats` times with a 300ms pause between runs.
    """
    # Exact frequencies from the original SDR (Capitol/EMI Canada, 1982)
    frequencies = [
        50, 100, 250, 400, 640, 1010, 1610,
        4000, 6350, 8100, #10100, # 12600, 15200, 18300
    ]
    burst_len = 0.18   # seconds per tone
    gap_len   = 0.023   # seconds of silence between tones

    # Set up PyAudio output
    p   = pyaudio.PyAudio()
    out = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        output=True,
        frames_per_buffer=OUTPUT_CHUNK,
    )

    def play_bytes(pcm: bytes):
        out.write(pcm)

    def silence(duration: float):
        n = int(rate * duration)
        play_bytes((np.zeros(n, dtype=np.float32) * 32767).astype(np.int16).tobytes())

    def tone_burst(freq: float, duration: float, vol: float):
        n    = int(rate * duration)
        t    = np.linspace(0, duration, n, False)
        tone = np.sin(2 * np.pi * freq * t) * vol
        fade = int(0.004 * rate)
        if fade > 0:
            tone[:fade]  *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        play_bytes((np.clip(tone, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    print(f"[alert] Playing XDR/SDR tone burst × {repeats} "
          f"(volume={volume:.2f}, rate={rate}Hz, ramp={ramp})")

    try:
        for rep in range(repeats):
            print(f"[alert] Run {rep + 1}/{repeats}...")

            for i, freq in enumerate(frequencies):
                # Optionally ramp volume up for higher frequencies to compensate
                # for hearing rolloff above ~10kHz — makes the sequence feel
                # more even in perceived loudness.
                if ramp:
                    vol = volume + (i / len(frequencies)) * 0.12
                    vol = min(vol, 1.0)
                else:
                    vol = volume

                print(f"  {freq:>6} Hz  vol={vol:.2f}")
                tone_burst(freq, burst_len, vol)
                silence(gap_len)

            # Final long tone at 15200Hz for 820ms (as per the original spec)
            #final_vol = min(volume + 0.12, 1.0) if ramp else volume
            #print(f"  15200 Hz  vol={final_vol:.2f}  (final long tone 820ms)")
            #tone_burst(15200, 0.820, final_vol)

            # Pause between repeats (not after the last one)
            if rep < repeats - 1:
                print(f"[alert] Pause between runs...")
                silence(0.3)

    finally:
        out.stop_stream()
        out.close()
        p.terminate()

    print("[alert] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test XDR/SDR cassette alert tone")
    parser.add_argument("--repeats", type=int,   default=2,    help="Number of runs (default: 2)")
    parser.add_argument("--volume",  type=float, default=0.85, help="Volume 0.0–1.0 (default: 0.85)")
    parser.add_argument("--rate",    type=int,   default=16000,help="Sample rate Hz (default: 16000)")
    parser.add_argument("--no-ramp", action="store_true",      help="Disable high-freq volume ramp")
    args = parser.parse_args()

    play_alert(
        repeats = args.repeats,
        volume  = args.volume,
        rate    = args.rate,
        ramp    = not args.no_ramp,
    )