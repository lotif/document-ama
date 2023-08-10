import logging
import sys


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)
sys.stdout = StreamToLogger(LOGGER, logging.INFO)
sys.stderr = StreamToLogger(LOGGER, logging.ERROR)
