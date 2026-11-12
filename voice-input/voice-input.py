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
import time

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
# Как часто искать (пере)подключённые устройства; он же таймаут select().
RESCAN_INTERVAL_S = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_devices(hotkey: str, skip=frozenset()) -> list[evdev.InputDevice]:
    """Return all input devices that expose the requested key.

    Пути из `skip` не открываются вовсе: это устройства, которые демон уже
    держит, и периодическое пересканирование не должно открывать их повторно.
    Неподходящие устройства закрываются, иначе каждый rescan терял бы их fd.
    """
    found = []
    for path in evdev.list_devices():
        if path in skip:
            continue
        try:
            dev = evdev.InputDevice(path)
        except Exception:
            continue
        try:
            caps = dev.capabilities(verbose=True)
            keys = [name for name, _ in caps.get(("EV_KEY", 1), [])]
        except Exception:
            keys = []
        if hotkey in keys:
            found.append(dev)
        else:
            dev.close()
    return found


class DeviceWatcher:
    """Подходящие устройства в селекторе: снятие отвалившихся и hotplug-rescan.

    Отвалившееся устройство (выдернули приёмник или ядро заново перечислило
    USB-хаб после выхода из сна) оставляет мёртвый fd: select() вечно считает
    его готовым, а read() бросает OSError. Его надо снять с селектора, иначе
    цикл крутится на 100% CPU. Замена появляется новым узлом
    /dev/input/eventN, его подхватывает rescan раз в RESCAN_INTERVAL_S.
    """

    def __init__(self, hotkey: str, discover=find_devices, clock=time.monotonic,
                 sleep=time.sleep, interval: float = RESCAN_INTERVAL_S):
        self.hotkey = hotkey
        self.discover = discover
        self.clock = clock
        self.sleep = sleep
        self.interval = interval
        self.selector = selectors_module.DefaultSelector()
        self.devices: dict[str, evdev.InputDevice] = {}  # path -> device
        self.last_scan = clock()

    def add(self, dev: evdev.InputDevice) -> None:
        self.selector.register(dev, selectors_module.EVENT_READ)
        self.devices[dev.path] = dev
        print(f"Found: {dev.path}  {dev.name}", flush=True)

    def drop(self, dev: evdev.InputDevice, reason: object) -> None:
        try:
            self.selector.unregister(dev)
        except (KeyError, ValueError):
            pass
        try:
            dev.close()
        except OSError:
            pass
        self.devices.pop(dev.path, None)
        print(f"Lost: {dev.path}  {dev.name} ({reason})", flush=True)

    def rescan(self) -> None:
        self.last_scan = self.clock()
        for dev in self.discover(self.hotkey, skip=set(self.devices)):
            self.add(dev)

    def wait(self) -> list[evdev.InputDevice]:
        """Ждать готовые устройства не дольше `interval`; по истечении — rescan."""
        if self.devices:
            ready = [key.fileobj for key, _ in self.selector.select(self.interval)]
        else:
            # Ждать нечего: спим явно, а не полагаемся на поведение select()
            # на пустом селекторе.
            self.sleep(self.interval)
            ready = []
        if self.clock() - self.last_scan >= self.interval:
            self.rescan()
        return ready


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
    time.sleep(0.3)
    subprocess.run(["xdotool", "key", "shift+Insert"], check=False)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class Recorder:
    """Состояние записи: идущий arecord, его wav и устройство, которое его начало."""

    def __init__(self, url: str, lang: str):
        self.url = url
        self.lang = lang
        self.proc: subprocess.Popen | None = None
        self.tmp_path: str | None = None
        self.dev_path: str | None = None

    def start(self, dev_path: str) -> None:
        self.tmp_path = tempfile.mktemp(suffix=".wav")
        self.proc = subprocess.Popen(
            [
                "arecord",
                "-f", "S16_LE",
                "-r", str(SAMPLE_RATE),
                "-c", "1",
                self.tmp_path,
            ],
            stderr=subprocess.DEVNULL,
        )
        self.dev_path = dev_path
        print("Recording...", flush=True)

    def finish(self) -> None:
        self.proc.terminate()
        self.proc.wait()
        self.proc = None
        print("Transcribing...", flush=True)

        url, lang = self.url, self.lang

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
            args=(self.tmp_path,),
            daemon=True,
        ).start()
        self.tmp_path = None
        self.dev_path = None

    def abort(self) -> None:
        """Остановить arecord и выбросить запись, ничего не транскрибируя."""
        self.proc.terminate()
        self.proc.wait()
        self.proc = None
        try:
            os.unlink(self.tmp_path)
        except FileNotFoundError:
            pass
        self.tmp_path = None
        self.dev_path = None
        print("Recording aborted: device lost", flush=True)


def handle_ready(watcher: DeviceWatcher, rec: Recorder, dev: evdev.InputDevice, hotkey: str) -> None:
    """Обработать события одного готового устройства (один шаг цикла)."""
    try:
        for event in dev.read():
            if event.type != evdev.ecodes.EV_KEY:
                continue
            kev = evdev.categorize(event)
            if kev.keycode != hotkey:
                continue

            if kev.keystate == evdev.events.KeyEvent.key_down and rec.proc is None:
                rec.start(dev.path)

            elif kev.keystate == evdev.events.KeyEvent.key_up and rec.proc is not None:
                rec.finish()

    except BlockingIOError:
        # Узел открыт с O_NONBLOCK: ложное пробуждение без событий — не отвал.
        pass
    except OSError as exc:
        watcher.drop(dev, exc)
        # Клавиша зажата на отвалившемся устройстве: key_up с него уже не придёт,
        # и arecord писал бы до следующего нажатия. Запись обрывается, а не
        # транскрибируется: после сна текст ушёл бы в окно, которое пользователь
        # уже не ждёт.
        if rec.proc is not None and rec.dev_path == dev.path:
            rec.abort()


def listen(watcher: DeviceWatcher, hotkey: str, url: str, lang: str) -> None:
    rec = Recorder(url, lang)

    print(f"Ready. Hold {hotkey} to record. Ctrl+C to quit.", flush=True)

    try:
        while True:
            for dev in watcher.wait():
                handle_ready(watcher, rec, dev, hotkey)
    except KeyboardInterrupt:
        if rec.proc:
            rec.proc.terminate()
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

    watcher = DeviceWatcher(args.key)
    watcher.rescan()
    if not watcher.devices:
        print(
            f"No device with {args.key} found.\n"
            "Check you are in the 'input' group: sudo usermod -a -G input $USER\n"
            "Then log out and back in.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check server is reachable
    try:
        requests.get(f"{args.url}/health", timeout=3).raise_for_status()
    except Exception as exc:
        print(f"Whisper server not reachable at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)

    listen(watcher, args.key, args.url, args.lang)


import selectors as selectors_module  # noqa: E402 (imported late to keep top-level clean)

if __name__ == "__main__":
    main()
