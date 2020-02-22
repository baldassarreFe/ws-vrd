import signal
from typing import Callable

from loguru import logger


class SigIntHandler(object):
    def __init__(self, handler: Callable):
        self.handler = handler
        self.old_handler = signal.signal(signal.SIGINT, self._internal_handler)

    def _internal_handler(self, sig, frame):
        signal.signal(signal.SIGINT, self.old_handler)
        logger.warning("Received SIGINT")
        self.handler()
        self.old_handler = signal.signal(signal.SIGINT, self._internal_handler)


class SigIntCatcher(SigIntHandler):
    def __init__(self):
        self._caught = False
        super().__init__(self._handler)

    def _handler(self):
        self._caught = True

    def __bool__(self):
        return self._caught
