import signal
import time
import wave
import numpy as np
import pyaudio

# -------------------------
# Match satellite_client constants exactly
# -------------------------
MIC_RATE        = 16000
SAMPLE_WIDTH    = 2
INPUT_CHANNELS  = 1
FORMAT          = pyaudio.paInt16
CHUNK           = 1024

INPUT_GAIN_TARGET_RMS  = 0.2
INPUT_GAIN_MAX_DB      = 12.0
INPUT_GAIN_NOISE_FLOOR = 0.02
# smooths the gain jumping: (if sluggish to respond, drop down)
INPUT_GAIN_SMOOTHING   = 0.97

OUTPUT_RAW       = "capture_raw.wav"
OUTPUT_PROCESSED = "capture_processed.wav"

# -------------------------
# Globals
# -------------------------
raw_frames       = []
processed_frames = []
running          = True
current_gain     = 1.0

def read_input(stream: pyaudio.Stream) -> tuple:
    global current_gain

    raw = stream.read(CHUNK, exception_on_overflow=False)

    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    raw_rms = float(np.sqrt(np.mean(arr ** 2)))

    if raw_rms >= INPUT_GAIN_NOISE_FLOOR:
        max_gain = 10.0 ** (INPUT_GAIN_MAX_DB / 20.0)
        target_gain = min(INPUT_GAIN_TARGET_RMS / raw_rms, max_gain)
    else:
        target_gain = current_gain  # hold during silence

    current_gain = (INPUT_GAIN_SMOOTHING * current_gain +
                   (1.0 - INPUT_GAIN_SMOOTHING) * target_gain)

    arr = np.clip(arr * current_gain, -1.0, 1.0)

    post_rms   = float(np.sqrt(np.mean(arr ** 2)))
    clip_count = int(np.sum(np.abs(arr) >= 0.999))
    processed  = (arr * 32767.0).astype(np.int16).tobytes()

    return raw, processed, raw_rms, current_gain, post_rms, clip_count

def save_wav(filename: str, frames: list, label: str):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(INPUT_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(MIC_RATE)
        wf.writeframes(b"".join(frames))
    duration = len(frames) * CHUNK / MIC_RATE
    print(f"[capture] {label}: saved {duration:.1f}s → {filename}")

def signal_handler(sig, frame):
    global running
    print("\n[capture] Ctrl+C received, saving...")
    running = False

def main():
    global running, raw_frames, processed_frames

    signal.signal(signal.SIGINT, signal_handler)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=INPUT_CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        start=False,
    )
    stream.start_stream()

    print("[capture] Flushing initial buffer...")
    for _ in range(4):
        stream.read(CHUNK, exception_on_overflow=False)

    print(f"[capture] Recording... Ctrl+C to stop and save.")
    print(f"[capture] {'Time':>6}  {'RawRMS':>8}  {'Gain':>7}  {'PostRMS':>8}  {'Clip':>5}")
    print("-" * 50)

    last_print  = time.time()
    total_clips = 0

    while running:
        raw, processed, raw_rms, gain, post_rms, clip_count = read_input(stream)
        raw_frames.append(raw)
        processed_frames.append(processed)
        total_clips += clip_count

        if time.time() - last_print >= 1.0:
            elapsed   = len(raw_frames) * CHUNK / MIC_RATE
            clip_warn = " !! CLIPPING !!" if clip_count > 0 else ""
            print(f"[capture] {elapsed:>5.1f}s  {raw_rms:>8.4f}  {gain:>7.2f}x  {post_rms:>8.4f}  {clip_count:>5}{clip_warn}")
            last_print = time.time()

    stream.close()
    p.terminate()

    if raw_frames:
        print(f"\n[capture] Total clipped samples: {total_clips}")
        save_wav(OUTPUT_RAW,       raw_frames,       "raw      ")
        save_wav(OUTPUT_PROCESSED, processed_frames, "processed")
    else:
        print("[capture] No audio captured.")

if __name__ == "__main__":
    main()