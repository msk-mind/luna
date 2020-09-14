import logging
import tempfile
import os


def init_logger():
    # Logging configuration
    log_location = os.path.join(tempfile.gettempdir(), 'log')
    if not os.path.exists(log_location):
        os.mkdir(log_location)

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    log_file = os.path.join(log_location, 'data-processing.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    print(">>>>>>>> log file at: " + log_file)
    return logger

