# Running the Voice Node on Startup (systemd)

These instructions set up `run_voice_node.sh` to start automatically at boot using a **systemd user service**. A user service is required (rather than a system service) because the voice node needs access to the user's audio session (PipeWire/PulseAudio). Running it as a system service will cause JACK/ALSA errors like:

```
jack server is not running or cannot be started
Expression 'alsa_snd_pcm_mmap_begin(...)' failed in 'src/hostapi/alsa/pa_linux_alsa.c'
```

## Prerequisites

- The repo is cloned to your home directory (this guide assumes `~/supernova_satellite_linux` — adjust paths if yours differs)
- The run script is executable:

  ```bash
  chmod +x ~/supernova_satellite_linux/run_voice_node.sh
  ```

## 1. Create the user service file

```bash
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/voice-node.service
```

Paste the following, replacing `jesse` with your username if different:

```ini
[Unit]
Description=Supernova Satellite Voice Node
After=network-online.target sound.target

[Service]
Type=simple
WorkingDirectory=/home/jesse/supernova_satellite_linux
ExecStart=/home/jesse/supernova_satellite_linux/run_voice_node.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

> **Note:** User services do not use a `User=` line, and none of the `systemctl --user` commands below should be run with `sudo`.

## 2. Enable and start the service

```bash
systemctl --user daemon-reload
systemctl --user enable --now voice-node.service
```

## 3. Enable lingering (start at boot without login)

By default, user services only run while you're logged in. To make the voice node start at boot with no login required:

```bash
sudo loginctl enable-linger $USER
```

This also ensures the user audio stack (PipeWire/PulseAudio) starts at boot, which the voice node depends on.

## 4. Verify

```bash
systemctl --user status voice-node.service
journalctl --user -u voice-node.service -f
```

If audio devices aren't found on the first attempt, reboot once — lingering user sessions sometimes need a fresh boot for the audio stack to initialize in the correct order.

## Managing the service

```bash
systemctl --user restart voice-node.service   # restart after config changes
systemctl --user stop voice-node.service      # stop
systemctl --user disable voice-node.service   # remove from startup
journalctl --user -u voice-node.service -f    # follow live logs
```

If you edit the `.service` file, run `systemctl --user daemon-reload` before restarting.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `status=203/EXEC` / `No such file or directory` | The `ExecStart` path doesn't match the actual file. Check for hyphen vs. underscore typos and verify with `ls -l <path>`. |
| `jack server is not running` / ALSA `mmap_begin` failures | The service is running outside the user audio session. Make sure you're using a **user** service (`~/.config/systemd/user/`), not a system one in `/etc/systemd/system/`. |
| Service doesn't start after reboot | Lingering isn't enabled. Run `sudo loginctl enable-linger $USER`. |
| Changes to the service file have no effect | Run `systemctl --user daemon-reload`, then restart the service. |