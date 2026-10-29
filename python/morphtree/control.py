"""Cooperative cancellation at candidate and boosting-iteration boundaries."""
import time

class SearchInterrupted(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

class RunControl:
    def __init__(self, cancel=None, timeout=None):
        if cancel is not None and not callable(cancel):
            raise ValueError('cancel must be callable or None')
        self.cancel = cancel
        self.deadline = None if timeout is None else time.monotonic() + timeout

    def check(self):
        if self.cancel is not None and self.cancel():
            raise SearchInterrupted('cancelled')
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchInterrupted('timeout')
