from data_processing.common.custom_logger import init_logger
from testfixtures import LogCapture
import pytest

@pytest.fixture()
def logger():
    logger = init_logger() 
    yield logger

def test_levels(logger):
    with LogCapture() as l:
        logger.info('a message')
        logger.warning('a warning')
        logger.error('an error')
        l.check(
            ('root', 'INFO', 'a message'),
            ('root', 'WARNING', 'a warning'),
            ('root', 'ERROR', 'an error'),
        )

def test_formatting(logger):
    with LogCapture() as l:
        logger.info("a message with string %s", "some output")
        l.check(('root', 'INFO', 'a message with string some output'))

def test_list(logger):
    with LogCapture() as l:
        logger.info([1,2,3])
        l.check(('root', 'INFO', '[1, 2, 3]'))

def test_dict(logger):
    with LogCapture() as l:
        logger.info({'list':[1,2,3], 'name':'foo'})
        l.check(('root', 'INFO', "{'list': [1, 2, 3], 'name': 'foo'}"))

# For some reason, this capturing no longer works
# def test_newlines(logger, caplog):
#     logger.info("some\nmultiline\noutput")
#     out = caplog.text
#     print ("out, err=", out)
#     # Make sure multi-lines appear with their own header, make sure to run with pytest -s
#     assert "root - INFO - output" in err
