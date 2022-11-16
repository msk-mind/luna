'''
Created on November 02, 2020

@author: pashaa@mskcc.org
'''
import timeit


class CodeTimer:
    def __init__(self, logger, name=None):
        self.logger = logger
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start)
        self.logger.info('Code block' + self.name + ' took: ' + str(self.took) + 's')
        if exc_type is not None:
            self.logger.exception("Exception raised during code execution:")