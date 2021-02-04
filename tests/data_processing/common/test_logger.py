from data_processing.common.custom_logger import init_logger
from testfixtures import LogCapture

def test_levels():
    with LogCapture() as l:
        logger = init_logger() 
        logger.info('a message')
        logger.warning('a warning')
        logger.error('an error')
        l.check(
            ('root', 'INFO', 'a message'),
            ('root', 'WARNING', 'a warning'),
            ('root', 'ERROR', 'an error'),
        )

def test_formatting():
    with LogCapture() as l:
        logger = init_logger() 
        logger.info("a message with string %s", "some output")
        l.check(('root', 'INFO', 'a message with string some output'))

def test_list():
    with LogCapture() as l:
        logger = init_logger() 
        logger.info([1,2,3])
        l.check(('root', 'INFO', '[1, 2, 3]'))

def test_dict():
    with LogCapture() as l:
        logger = init_logger() 
        logger.info({'list':[1,2,3], 'name':'foo'})
        l.check(('root', 'INFO', "{'list': [1, 2, 3], 'name': 'foo'}"))

def test_newlines(capfd):
    logger = init_logger() 
    logger.info("some\nmultiline\noutput")
    out, err = capfd.readouterr()
    # Make sure multi-lines appear with their own header, make sure to run with pytest -s
    assert "root - INFO - output" in err
