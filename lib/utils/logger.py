import sys
import logging


def get_logger(logfile_path, logger_name=None):
    # Initialize logger.
    logFormatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(logger_name)

    fileHandler = logging.FileHandler(logfile_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)

    return logger

