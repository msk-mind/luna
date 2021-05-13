import logging

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


def init_logger(filename='data-processing.log', level=logging.WARNING):
    # Logging configuration
    log_file = filename

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = MultilineFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        # create console handler with a customizable, higher log level
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # create file handler which logs even debug messages
        fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=1e7, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    print(">>>>>>>> Initalized logger, log file at: " + log_file + " with handlers: " + str(logger.handlers))
    return logger



