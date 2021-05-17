import os, logging
from logging.handlers import RotatingFileHandler
from log4mongo.handlers import MongoHandler

from data_processing.common.config import ConfigSet

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
    formatter = MultilineFormatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    if os.path.exists('conf/logging.cfg'):
        cfg = ConfigSet(name='LOG_CFG',  config_file='conf/logging.cfg')
    else:
        cfg = ConfigSet(name='LOG_CFG',  config_file='conf/logging.default.yml')
    
    if not logger.handlers:
        # create console handler with a customizable, higher log level
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # create file handler which logs even debug messages
        fh = RotatingFileHandler(log_file, maxBytes=1e7, backupCount=10)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # create mongo log handler if we configured it
        if cfg and cfg.get_value('LOG_CFG::CENTRAL_LOGGING'):
            mh = MongoHandler (
                host=cfg.get_value('LOG_CFG::MONGO_HOST'), 
                port=cfg.get_value('LOG_CFG::MONGO_PORT'), 
                capped=True )
            mh.setLevel(logging.WARNING)
            logger.addHandler(mh)
    
    logger.info("FYI: Initalized logger, log file at: " + log_file + " with handlers: " + str(logger.handlers))
    return logger



