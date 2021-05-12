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


def init_logger(filename='data-processing.log', level=logging.INFO):
    # Logging configuration
    log_file = filename

    logger = logging.getLogger()
    formatter = MultilineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # fh = logging.FileHandler(log_file)
    # fh.setLevel(level)
    # fh.setFormatter(formatter)
    
    if not logger.handlers:
        logger.setLevel(level)
        # create file handler which logs even debug messages
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(level)
        # create formatter and add it to the handlers
        ch.setFormatter(formatter)
    #     # add the handlers to logger
        logger.addHandler(ch)
    #     logger.addHandler(fh)
    # if logger.handlers:
    #     logger.addHandler(fh)
    print(">>>>>>>> Initalized logger, log file at: " + log_file + " with handlers: " + str(logger.handlers))
    return logger



