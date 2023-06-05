import os

import pytest
from testfixtures.logcapture import LogCapture

from luna.common.custom_logger import init_logger


@pytest.fixture()
def logger():
    os.environ["LUNA_HOME"] = "tests/testdata/common/"
    logger = init_logger()
    yield logger


def test_no_luna_home():
    os.environ["LUNA_HOME"] = ""
    with pytest.raises(RuntimeError):
        init_logger()


def test_check_level(logger):
    # 20 for info. ref: https://docs.python.org/3/library/logging.html#levels
    assert 20 == logger.getEffectiveLevel()


def test_levels(logger):
    with LogCapture() as log:
        logger.info("a message")
        logger.warning("a warning")
        logger.error("an error")
        log.check(
            ("root", "INFO", "a message"),
            ("root", "WARNING", "a warning"),
            ("root", "ERROR", "an error"),
        )


def test_formatting(logger):
    with LogCapture() as log:
        logger.info("a message with string %s", "some output")
        log.check(("root", "INFO", "a message with string some output"))


def test_list(logger):
    with LogCapture() as log:
        logger.info([1, 2, 3])
        log.check(("root", "INFO", "[1, 2, 3]"))


def test_dict(logger):
    with LogCapture() as log:
        logger.info({"list": [1, 2, 3], "name": "foo"})
        log.check(("root", "INFO", "{'list': [1, 2, 3], 'name': 'foo'}"))


# For some reason, this capturing no longer works
# def test_newlines(logger, caplog):
#     logger.info("some\nmultiline\noutput")
#     out = caplog.text
#     print ("out, err=", out)
#     # Make sure multi-lines appear with their own header, make sure to run with pytest -s
#     assert "root - INFO - output" in err