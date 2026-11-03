import threading
import time

from drivers.base import StallWatchdog


def test_watchdog_zeroes_after_stall_and_rearms_on_feed():
    tripped = threading.Event()
    dog = StallWatchdog(tripped.set, timeout_s=0.15)
    dog.start()
    try:
        for _ in range(5):  # fed regularly -> never trips
            dog.feed()
            time.sleep(0.05)
        assert not tripped.is_set()

        assert tripped.wait(0.5)  # stop feeding -> trips
        assert dog.tripped
        dog.feed()
        assert not dog.tripped
    finally:
        dog.stop()
