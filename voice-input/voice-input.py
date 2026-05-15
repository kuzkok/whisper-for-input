#!/usr/bin/env python3
"""
Push-to-talk voice input for KDE Wayland.

Hold HOTKEY → records audio → Whisper transcribes → pastes into active window.

Requirements (host):
    pip install evdev requests
    sudo dnf install alsa-utils ydotool wl-clipboard
    sudo usermod -a -G input $USER   # then re-login

Usage:
    python3 voice-input.py [--key KEY_SCROLLLOCK] [--lang ru] [--url http://localhost:8000]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading

import evdev
import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_HOTKEY = "KEY_SCROLLLOCK"
# Empty string = auto-detect (best for mixed Russian/English speech).
# Override with e.g. --lang ru or --lang en if you need strict single-language mode.
DEFAULT_LANG = ""
DEFAULT_URL = "http://localhost:8000"
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_devices(hotkey: str) -> list[evdev.InputDevice]:
    """Return all input devices that expose the requested key."""
    found = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities(verbose=True)
            keys = [name for name, _ in caps.get(("EV_KEY", 1), [])]
            if hotkey in keys:
                found.append(dev)
        except Exception:
            pass
    return found


def transcribe(wav_path: str, url: str, lang: str) -> str:
    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{url}/transcribe",
            files={"file": ("audio.wav", f, "audio/wav")},
            data={"language": lang},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["text"]


def type_text(text: str) -> None:
    if not text:
        return
    # Copy to Wayland clipboard, then simulate Ctrl+V.
    # More reliable than ydotool type for non-ASCII (Cyrillic etc.).
    subprocess.run(["wl-copy", text], check=False)
    import time; time.sleep(0.3)
    subprocess.run(["xdotool", "key", "shift+Insert"], check=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def listen(devices: list[evdev.InputDevice], hotkey: str, url: str, lang: str) -> None:
    record_proc: subprocess.Popen | None = None
    tmp_path: str | None = None

    # Merge events from all matching devices
    selector = selectors_module.DefaultSelector()
    for dev in devices:
        selector.register(dev, selectors_module.EVENT_READ)

    print(f"Ready. Hold {hotkey} to record. Ctrl+C to quit.", flush=True)

    try:
        while True:
            for key, _ in selector.select():
                dev: evdev.InputDevice = key.fileobj
                try:
                    for event in dev.read():
                        if event.type != evdev.ecodes.EV_KEY:
                            continue
                        kev = evdev.categorize(event)
                        if kev.keycode != hotkey:
                            continue

                        if kev.keystate == evdev.events.KeyEvent.key_down and record_proc is None:
                            tmp_path = tempfile.mktemp(suffix=".wav")
                            record_proc = subprocess.Popen(
                                [
                                    "arecord",
                                    "-f", "S16_LE",
                                    "-r", str(SAMPLE_RATE),
                                    "-c", "1",
                                    tmp_path,
                                ],
                                stderr=subprocess.DEVNULL,
                            )
                            print("Recording...", flush=True)

                        elif kev.keystate == evdev.events.KeyEvent.key_up and record_proc is not None:
                            record_proc.terminate()
                            record_proc.wait()
                            record_proc = None
                            print("Transcribing...", flush=True)

                            # Run in a thread so we don't block the event loop
                            def _transcribe_and_type(path):
                                try:
                                    text = transcribe(path, url, lang)
                                    print(f"→ {text}", flush=True)
                                    type_text(text)
                                except Exception as exc:
                                    print(f"Error: {exc}", file=sys.stderr, flush=True)
                                finally:
                                    try:
                                        os.unlink(path)
                                    except FileNotFoundError:
                                        pass

                            threading.Thread(
                                target=_transcribe_and_type,
                                args=(tmp_path,),
                                daemon=True,
                            ).start()
                            tmp_path = None

                except OSError:
                    pass
    except KeyboardInterrupt:
        if record_proc:
            record_proc.terminate()
        print("\nBye.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Push-to-talk Whisper input")
    parser.add_argument("--key", default=DEFAULT_HOTKEY, help=f"evdev key name (default: {DEFAULT_HOTKEY})")
    parser.add_argument("--lang", default=DEFAULT_LANG, help=f"language code (default: {DEFAULT_LANG})")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Whisper server URL (default: {DEFAULT_URL})")
    parser.add_argument("--list-keys", action="store_true", help="list available key names and exit")
    args = parser.parse_args()

    if args.list_keys:
        print("\n".join(k for k in dir(evdev.ecodes) if k.startswith("KEY_")))
        return

    devices = find_devices(args.key)
    if not devices:
        print(
            f"No device with {args.key} found.\n"
            "Check you are in the 'input' group: sudo usermod -a -G input $USER\n"
            "Then log out and back in.",
            file=sys.stderr,
        )
        sys.exit(1)

    for dev in devices:
        print(f"Found: {dev.path}  {dev.name}")

    # Check server is reachable
    try:
        requests.get(f"{args.url}/health", timeout=3).raise_for_status()
    except Exception as exc:
        print(f"Whisper server not reachable at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    listen(devices, args.key, args.url, args.lang)


import selectors as selectors_module  # noqa: E402 (imported late to keep top-level clean)

if __name__ == "__main__":
    main()
