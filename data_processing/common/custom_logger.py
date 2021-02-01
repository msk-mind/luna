import logging
import tempfile
import os

class MultilineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        save_msg = str(record.msg)
        output = ""
        for line in save_msg.splitlines():
            record.msg = line
            output += super().format(record) + "\n"
        output = output.rstrip()
        record.msg     = save_msg
        record.message = output
        return output


def init_logger(filename='data-processing.log', logname='root'):
    # Logging configuration
    log_file = filename

    logger = logging.getLogger(logname)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = MultilineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    print(">>>>>>>> log file at: " + log_file)
    return logger

