'''Module for general logger.

'''

import logging


logger = logging.getLogger('cortex')
logger.setLevel(logging.DEBUG)
logger.propagate = False
file_formatter = logging.Formatter(
    '%(asctime)s:%(name)s[%(levelname)s]:%(message)s')
stream_formatter = logging.Formatter(
    '[%(levelname)s]:%(message)s')

def set_stream_logger():
    global logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    logger.info('Saving logs to %s' % file_path)
