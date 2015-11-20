'''
Module for general logger.
'''

import logging

loggers = {}

def setup_custom_logger(name, level):
    global loggers

    logger = loggers.get(name, None)
    if logger is not None:
        logger.setLevel(level)
        return logger
    formatter = logging.Formatter(fmt='%(asctime)s:%(levelname)s:'
                                  '%(module)s:%(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    loggers[name] = logger
    return logger