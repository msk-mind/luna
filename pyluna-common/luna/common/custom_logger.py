import os
import logging
from logging.handlers import RotatingFileHandler
from log4mongo.handlers import MongoHandler

from luna.common.config import ConfigSet


class MultilineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        save_msg = str(record.msg)
        output = ""
        for line in save_msg.splitlines():
            record.msg = line
            output += super().format(record) + "\n"
        output = output.rstrip()
        record.msg = save_msg
        record.message = output
        return output


def init_logger(filename="data-processing.log"):
    # Logging configuration
    if os.environ["LUNA_HOME"]:
        cfg = ConfigSet(
            name="LOG_CFG",
            config_file=os.path.join(os.environ["LUNA_HOME"], "conf", "logging.cfg"),
        )
    else:
        raise RuntimeError(
            "$LUNA_HOME is not set. Make sure you have set $LUNA_HOME and $LUNA_HOME/conf/logging.cfg"
        )

    log_file = filename
    logger = logging.getLogger()
    logger.setLevel(cfg.get_value("LOG_CFG::LOG_LEVEL"))
    formatter = MultilineFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if not logger.handlers:
        # create console handler with a customizable, higher log level
        ch = logging.StreamHandler()
        ch.setLevel(cfg.get_value("LOG_CFG::LOG_LEVEL"))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # create file handler which logs even debug messages
        fh = RotatingFileHandler(log_file, maxBytes=1e7, backupCount=10)
        fh.setLevel(cfg.get_value("LOG_CFG::LOG_LEVEL"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # create mongo log handler if we configured it
        if cfg and cfg.get_value("LOG_CFG::CENTRAL_LOGGING"):
            mh = MongoHandler(
                host=cfg.get_value("LOG_CFG::MONGO_HOST"),
                port=cfg.get_value("LOG_CFG::MONGO_PORT"),
                capped=True,
            )
            mh.setLevel(cfg.get_value("LOG_CFG::MONGO_LOG_LEVEL"))
            logger.addHandler(mh)

    logger.info("Initalized logger, log file at: " + log_file)
    logger.debug("Initalized logger with handlers: " + str(logger.handlers))
    return logger
