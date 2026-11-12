"""Тесты цикла событий voice-input на фейковых устройствах.

Реальный evdev не трогаем: до импорта модуля в sys.modules кладётся заглушка
`evdev`, а устройства — объекты на os.pipe(), чтобы настоящий selectors видел
настоящий fd. «Мёртвое» устройство — pipe с непрочитанным байтом: select()
всегда считает его готовым, а read() бросает OSError, ровно как узел
/dev/input/eventN после отвала USB-хаба.

Запуск: cd voice-input && python3 -m unittest discover -s tests
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

EV_KEY = 1
KEY_DOWN, KEY_UP = 1, 0
HOTKEY = "KEY_SCROLLLOCK"


def _install_stubs():
    evdev = types.ModuleType("evdev")
    evdev.ecodes = types.SimpleNamespace(EV_KEY=EV_KEY)
    evdev.events = types.SimpleNamespace(
        KeyEvent=types.SimpleNamespace(key_down=KEY_DOWN, key_up=KEY_UP)
    )
    evdev.categorize = lambda event: event
    evdev.list_devices = lambda: []

    def _no_real_devices(path):
        raise AssertionError(f"тест попытался открыть реальное устройство {path}")

    evdev.InputDevice = _no_real_devices
    sys.modules["evdev"] = evdev
    if "requests" not in sys.modules:
        try:
            import requests  # noqa: F401
        except ImportError:
            sys.modules["requests"] = types.ModuleType("requests")


_install_stubs()
_spec = importlib.util.spec_from_file_location(
    "voice_input", Path(__file__).resolve().parent.parent / "voice-input.py"
)
vi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vi)


def key_event(state, keycode=HOTKEY):
    return types.SimpleNamespace(type=EV_KEY, keycode=keycode, keystate=state)


class FakeDevice:
    """Устройство на pipe: read() отдаёт очередь событий или бросает ошибку."""

    def __init__(self, path, name="fake keyboard"):
        self.path = path
        self.name = name
        self._r, self._w = os.pipe()
        self.batches = []  # что вернуть следующими вызовами read(): list | Exception
        self.reads = 0
        self.closed = False

    def fileno(self):
        return self._r

    def push(self, batch):
        """Поставить ответ read() в очередь и сделать fd готовым к чтению."""
        self.batches.append(batch)
        os.write(self._w, b"x")

    def kill(self):
        """Устройство отвалилось: fd всегда готов, read() всегда OSError."""
        self.batches = None
        os.write(self._w, b"x")

    def read(self):
        self.reads += 1
        if self.reads > 50:
            # Страховка: на старом коде цикл крутился бы тут вечно.
            raise KeyboardInterrupt("busy-loop: read() зовётся без конца")
        if self.batches is None:
            raise OSError(19, "No such device")
        os.read(self._r, 1)
        batch = self.batches.pop(0)
        if isinstance(batch, BaseException):
            raise batch
        return iter(batch)

    def close(self):
        if not self.closed:
            self.closed = True
            os.close(self._r)
            os.close(self._w)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Harness:
    """DeviceWatcher с подменённым discover, часами и sleep."""

    def __init__(self, test, max_sleeps=3):
        self.available = []  # что «видно» в /dev/input при следующем пересканировании
        self.discover_calls = []
        self.sleeps = []
        self.clock = FakeClock()
        self.max_sleeps = max_sleeps
        self.watcher = vi.DeviceWatcher(
            HOTKEY,
            discover=self.discover,
            clock=self.clock,
            sleep=self.sleep,
        )
        # Настоящий select(), но без настоящего ожидания: пустой результат
        # засчитывается как истёкший таймаут — часы идут, лимит ожиданий тикает.
        real_select = self.watcher.selector.select
        self.selects = []

        def select(timeout=None):
            self.selects.append(timeout)
            ready = real_select(0)
            if not ready:
                self.sleep(timeout)
            return ready

        self.watcher.selector.select = select
        test.addCleanup(self.cleanup)

    def discover(self, hotkey, skip=frozenset()):
        self.discover_calls.append(set(skip))
        found = [d for d in self.available if d.path not in skip]
        self.available = [d for d in self.available if d.path in skip]
        return found

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock.now += seconds
        if len(self.sleeps) >= self.max_sleeps:
            raise KeyboardInterrupt

    def cleanup(self):
        for dev in list(self.watcher.devices.values()):
            dev.close()
        self.watcher.selector.close()


class DropTests(unittest.TestCase):
    def test_oserror_drops_device_without_busy_loop(self):
        h = Harness(self, max_sleeps=1)
        dead = FakeDevice("/dev/input/event5")
        self.addCleanup(dead.close)
        h.watcher.add(dead)
        dead.kill()

        vi.listen(h.watcher, HOTKEY, "http://unused", "")

        self.assertEqual(dead.reads, 1)
        self.assertTrue(dead.closed)
        self.assertEqual(h.watcher.devices, {})
        self.assertEqual(len(h.watcher.selector.get_map()), 0)
        # После потери последнего устройства цикл ушёл в sleep, а не в select.
        self.assertEqual(h.sleeps, [vi.RESCAN_INTERVAL_S])

    def test_blocking_io_error_does_not_drop_device(self):
        # evdev открывает узел с O_NONBLOCK: ложное пробуждение даёт EAGAIN,
        # это не отвал устройства.
        h = Harness(self)
        dev = FakeDevice("/dev/input/event3")
        self.addCleanup(dev.close)
        h.watcher.add(dev)
        dev.push(BlockingIOError(11, "Resource temporarily unavailable"))

        self.assertEqual(h.watcher.wait(), [dev])
        state = vi.Recorder("http://unused", "")
        vi.handle_ready(h.watcher, state, dev, HOTKEY)

        self.assertFalse(dev.closed)
        self.assertIn(dev.path, h.watcher.devices)


class RescanTests(unittest.TestCase):
    def test_rescan_picks_up_reconnected_device(self):
        h = Harness(self, max_sleeps=2)
        old = FakeDevice("/dev/input/event5")
        self.addCleanup(old.close)
        h.watcher.add(old)
        h.watcher.rescan()
        old.kill()
        new = FakeDevice("/dev/input/event9")
        self.addCleanup(new.close)
        h.available = [new]

        vi.listen(h.watcher, HOTKEY, "http://unused", "")

        self.assertTrue(old.closed)
        self.assertEqual(list(h.watcher.devices), ["/dev/input/event9"])
        registered = [key.fileobj for key in h.watcher.selector.get_map().values()]
        self.assertEqual(registered, [new])

    def test_rescan_happens_only_after_interval(self):
        h = Harness(self)
        dev = FakeDevice("/dev/input/event3")
        self.addCleanup(dev.close)
        h.watcher.add(dev)
        h.watcher.rescan()
        calls = len(h.discover_calls)

        dev.push([])
        h.watcher.wait()
        self.assertEqual(len(h.discover_calls), calls)

        h.clock.now += vi.RESCAN_INTERVAL_S
        dev.push([])
        h.watcher.wait()
        self.assertEqual(len(h.discover_calls), calls + 1)

    def test_open_device_is_not_reopened(self):
        h = Harness(self)
        dev = FakeDevice("/dev/input/event3")
        self.addCleanup(dev.close)
        h.watcher.add(dev)

        h.watcher.rescan()

        self.assertEqual(h.discover_calls[-1], {"/dev/input/event3"})


class FindDevicesTests(unittest.TestCase):
    def test_skips_open_paths_and_closes_non_matching(self):
        opened = {}

        class Dev:
            def __init__(self, path):
                opened[path] = self
                self.path = path
                self.closed = False

            def capabilities(self, verbose=False):
                keys = [(HOTKEY, 70)] if self.path.endswith("1") else [("KEY_A", 30)]
                return {("EV_KEY", 1): keys}

            def close(self):
                self.closed = True

        paths = ["/dev/input/event0", "/dev/input/event1", "/dev/input/event2"]
        with mock.patch.object(vi.evdev, "list_devices", lambda: paths), \
                mock.patch.object(vi.evdev, "InputDevice", Dev):
            found = vi.find_devices(HOTKEY, skip={"/dev/input/event2"})

        self.assertEqual([d.path for d in found], ["/dev/input/event1"])
        self.assertNotIn("/dev/input/event2", opened)
        self.assertTrue(opened["/dev/input/event0"].closed)
        self.assertFalse(opened["/dev/input/event1"].closed)


class NoDevicesTests(unittest.TestCase):
    def test_no_devices_waits_instead_of_spinning(self):
        h = Harness(self, max_sleeps=3)
        h.watcher.rescan()

        with mock.patch.object(h.watcher.selector, "select") as select:
            vi.listen(h.watcher, HOTKEY, "http://unused", "")

        select.assert_not_called()
        self.assertEqual(h.sleeps, [vi.RESCAN_INTERVAL_S] * 3)
        # Каждый таймаут — попытка найти устройство заново.
        self.assertEqual(len(h.discover_calls), 1 + 2)


class RecordingTests(unittest.TestCase):
    def test_device_lost_while_recording_aborts_arecord(self):
        h = Harness(self, max_sleeps=1)
        dev = FakeDevice("/dev/input/event5")
        self.addCleanup(dev.close)
        h.watcher.add(dev)
        dev.push([key_event(KEY_DOWN)])
        procs = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                self.path = args[-1]
                Path(self.path).write_bytes(b"RIFF")
                self.terminated = False
                procs.append(self)

            def terminate(self):
                self.terminated = True

            def wait(self):
                return 0

        def first_wait():
            # Первый read() отдал key_down — теперь, с зажатой клавишей,
            # устройство отваливается.
            if dev.reads == 1 and dev.batches is not None:
                dev.kill()
            return orig_wait()

        orig_wait = h.watcher.wait
        with mock.patch.object(vi.subprocess, "Popen", FakePopen), \
                mock.patch.object(vi.threading, "Thread") as thread, \
                mock.patch.object(h.watcher, "wait", first_wait):
            vi.listen(h.watcher, HOTKEY, "http://unused", "")

        self.assertEqual(len(procs), 1)
        self.assertTrue(procs[0].terminated)
        self.assertFalse(Path(procs[0].path).exists())
        thread.assert_not_called()  # обрывок записи не транскрибируем
        self.assertTrue(dev.closed)

    def test_hotkey_press_release_still_transcribes(self):
        h = Harness(self, max_sleeps=1)
        dev = FakeDevice("/dev/input/event3")
        self.addCleanup(dev.close)
        h.watcher.add(dev)
        dev.push([key_event(KEY_DOWN), key_event(KEY_UP)])
        procs = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                self.terminated = False
                procs.append(self)

            def terminate(self):
                self.terminated = True

            def wait(self):
                return 0

        sent = []
        real_wait = h.watcher.wait

        def wait_then_stop():
            if sent:
                raise KeyboardInterrupt
            sent.append(1)
            return real_wait()

        with mock.patch.object(vi.subprocess, "Popen", FakePopen), \
                mock.patch.object(vi.threading, "Thread") as thread, \
                mock.patch.object(h.watcher, "wait", wait_then_stop):
            vi.listen(h.watcher, HOTKEY, "http://unused", "")

        self.assertEqual(len(procs), 1)
        self.assertTrue(procs[0].terminated)
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()
        self.assertFalse(dev.closed)


if __name__ == "__main__":
    unittest.main()
