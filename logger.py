import logging
import pathlib
from datetime import datetime
import sys
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

def set_logging_path(path):
	parent_path = pathlib.Path(path).parent
	parent_path.mkdir(parents=True, exist_ok=True)

	logger.handlers = []

	h = logging.StreamHandler(sys.stderr)
	f = logging.Formatter('%(message)s')
	h.setFormatter(f)
	logger.addHandler(h)

	h = logging.FileHandler(path)
	f = logging.Formatter("%(asctime)s: %(message)s")
	h.setFormatter(f)
	logger.addHandler(h)