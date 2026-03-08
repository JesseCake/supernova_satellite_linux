
# Supernova Satellite (Linux)

A compact voice "satellite" client that listens for a wake word, then opens a streaming session to a voice server for ASR/TT S. Designed for lightweight Linux devices (Jetson / Raspberry Pi / desktop) using PyAudio and OpenWakeWord. Intended to be linked to my other project "supernova" [(http)](https://github.com/JesseCake/supernova) to the "voice_remote" interface.

This repository contains a working client (`satellite_client.py`). This implements: detect a wake word locally, handshake with a remote voice server, stream microphone audio, and play back server TTS.

For now there is a `dead_satellite_client.py` file that was the previous code. I leaned heavily on Claude to refactor this as it had gotten out of control (I've only been tinkering with this in my spare time, with large gaps between)

## Quick overview

- Wait for a wake word (OpenWakeWord model, e.g. `soopanova.onnx`).
- When wake is detected, connect to the configured voice server and send a `WAKE` frame.
- The server can send TTS audio and control frames (RDY0, THNK, CLOS, etc.). The client plays incoming audio and, after the server signals it's ready, opens the microphone and streams `AUD0` frames to the server until the server closes the session.

The project is deliberately small and easy to run on headless Linux systems. Audio I/O uses PyAudio (ALSA/Pulse), and wake-word detection uses the `openwakeword` Python package.

## Files you care about

- `satellite_client.py` — main, production-style client. Reads `config.yaml` and runs the wake/stream loop.
- `config.yaml` — simple configuration (voice server host/port and wake-word settings).
- `run_voice_node.sh` — helper script to activate a venv and run the client.
- `requirements.txt` — Python dependencies.
- `*.onnx` — pre-built wake-word models included for convenience (e.g. `soopanova.onnx`).

## Requirements

- Linux (ALSA/Pulse compatible audio stack recommended).
- Python 3.8+ (a virtualenv is recommended).
- System packages required by PyAudio (on Debian/Ubuntu: `portaudio19-dev`, `python3-dev`, etc.).
- Python deps: see `requirements.txt`:
	- numpy
	- openwakeword (tested with 0.4.0 in this repo)
	- pyyaml
	- pyaudio

Example: on Debian/Ubuntu-based systems you might need to install:

```bash
sudo apt update
sudo apt install -y build-essential portaudio19-dev python3-dev pulseaudio
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: PyAudio can be finicky depending on distro — using distribution packages for PortAudio and Python headers avoids most build issues.

## Configuration

Edit `config.yaml` to point at your voice server and to choose (or fine-tune) the wake-word model:

```yaml
voice_server:
	host: "jetson.lan"
	port: 10400
wake_word:
	model_path: "./soopanova.onnx"
	threshold: 0.005   # lower = easier to trigger, raise to reduce false positives
	cooldown_s: 0.01   # cooldown between accepted triggers
```

- `model_path` should be a path to a wake-word model in ONNX format.
- `threshold` controls the score required to accept a wake. If you get many false triggers, raise it; if it never triggers, lower it.
- `cooldown_s` prevents rapid retriggers immediately after a detection.

## Running

1. Create and activate a venv and install requirements (see commands above).
2. Make sure `config.yaml` points at a reachable voice server.
3. Run the client:

```bash
# Activate your venv if you didn't already
source .venv/bin/activate
python satellite_client.py
```

Or use the helper script (expects `.venv` next to the script):

```bash
./run_voice_node.sh
```

## How it works — simplified logic flow

1. Startup: audio output is opened and a short two-tone beep plays to indicate the client is live.
2. Wake-word loop (State.IDLE): the client opens the microphone with a frame size tuned to the OpenWakeWord model and repeatedly calls `openwakeword.Model.predict()` on short frames.
3. When a wakeword score exceeds the configured threshold, the client:
	 - plays a confirmation beep,
	 - connects to the voice server and sends a `WAKE` frame,
	 - enters a session state where it spawns receiver and microphone-sender threads.
4. Server-to-client frames (examples):
	 - `TTS0` — raw PCM int16 16 kHz audio to play (client enqueues and plays it),
	 - `BEEP` — short beep audio,
	 - `RDY0` — server indicates it has finished sending any initial TTS and the client can open the mic,
	 - `THNK` — server is processing (client can play a short "thinking" beep),
	 - `CLOS` — session is closing.
5. Once the server sends `RDY0` and any playback has drained locally, the client opens the mic in streamer mode and begins sending `AUD0` frames containing mic PCM until the server closes the session.
6. After `CLOS`, the client plays closing beeps, tears down the socket, flushes the wake-word model state with a short period of silence, and returns to the wake-word loop.

The client uses a small state machine with these states: IDLE → WAITING → LISTENING → THINKING → SPEAKING → CLOSING. Transitions are typically driven by wake detection and incoming server frames.

## Protocol (frame-level)

The client and server exchange framed messages. Each frame contains a 4-byte ASCII tag and a length-prefixed payload. Key tags used by the client:

- WAKE — sent by client to initiate a session after a wake word.
- AUD0 — client → server: raw microphone PCM (int16, 16 kHz mono) chunks.
- TTS0 — server → client: raw PCM to be played back (int16, 16 kHz mono).
- BEEP — server → client: short beep audio to play locally.
- RDY0 — server → client: indicates server is ready for mic audio (after sending any initial TTS).
- THNK — server → client: processing indicator (client plays a short beep).
- CLOS — server → client: close session.

Frames are packed/unpacked using a simple struct header matching the code in `satellite_client.py` (`<4sI`, tag + uint32 length).

## Training your own OpenWakeWord model (high level)

You can use the included ONNX models, but if you want a custom wake phrase tuned to your voice or environment you can train a new model with OpenWakeWord. The exact training commands depend on the OpenWakeWord release and tooling; below are safe, practical steps and recommendations:

1. Collect data
	 - Positive examples: 30–300 recordings of the wake phrase, from different speakers and microphones if possible. Keep clips short (a single phrase each, 0.5–2s).
	 - Negative examples: a larger set of environmental and speech clips that do not contain the wake phrase (minutes to hours of varied audio).
	 - Use WAV, 16 kHz, 16-bit mono.

2. Organize dataset
	 - A simple structure is:

		 dataset/
			 positives/
				 pos_000.wav
				 pos_001.wav
			 negatives/
				 neg_000.wav
				 ...

3. Use OpenWakeWord training tools
	 - Follow the OpenWakeWord project's training documentation (its repository contains scripts and a trainer). Typical steps are:
		 - Convert audio to mel-spectrogram features used by OpenWakeWord.
		 - Train a lightweight model (PyTorch/ONNX exportable).
		 - Export the trained model to ONNX and copy it into this project (e.g. `./my_wake.onnx`).

4. Update `config.yaml`
	 - Point `wake_word.model_path` to your new ONNX file and tune `threshold`.

Notes and tips
	- If your OpenWakeWord install is missing resource files, the client calls `openwakeword.utils.download_models()` automatically to fetch required assets (e.g. `melspectrogram.onnx`).
	- If you only need a quick experiment, try fine-tuning threshold first before training a new model.
	- When training: validate the model on held-out positive and negative clips and inspect ROC/precision-recall to choose a reasonable operating threshold.

If you want a concrete set of commands to train and export a model, tell me which OpenWakeWord version/tooling you plan to use (or point me to the repo/docs you want to follow) and I will produce a step-by-step recipe with explicit CLI or Python commands.

## Tuning & troubleshooting

- No wake triggers: lower `wake_word.threshold`, ensure mic is open and devices are correct, ensure `openwakeword` resources are present.
- Too many false positives: raise `threshold`, add more negative training data, or increase `cooldown_s`.
- No audio output: check Pulse/ALSA, verify PyAudio can open output. Try listing devices with the example `list_devices()` helper in `dead_satellite_client.py`.
- Missing `melspectrogram.onnx` or other resources: the client will attempt to download them; check network access or install the `openwakeword` resource bundle manually.

## Developer notes

- The repository intentionally keeps network protocol and audio formats simple (raw int16 PCM @ 16 kHz mono). This keeps server implementations language-agnostic.
- The wake-word loop uses small hops tuned to common OpenWakeWord models (1280-sample hops at 16 kHz).

## Who wrote this?

This has been the work of Jesse Stevens in tandem with a heap of "vibe-coding" primarily through Github Copilot, however some input from ChatGPT (no longer using), and then a recent refactor from Claude by Anthropic. Speech processing is not my forte, so I use these tools openly and transparently to get me from A to B. The important part here is that we have the power to create voice assistants at home that run completely locally. 

## License

This project is licensed under the MIT License — see the accompanying `LICENSE` file for the full terms. I believe strongly in open source, and the world will only advance if we all do the same. Please use this as you please and do great things with it!

