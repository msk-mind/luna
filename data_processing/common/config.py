'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''

import yaml
import sys

from data_processing.common.custom_logger import init_logger

logger = init_logger()

class Config():

    def __init__(self, config_file="config.yaml"):
        self._config_file = config_file


    def _get_config(self):
        '''

        :param config_file: Default config file is config.yaml
        :return: config generator object
        '''
        # read config file
        try:
            stream = open(self._config_file, 'r')
        except IOError as err:
            logger.error("unable to find a config file with name "+self._config_file+
                  ". Please use config.yaml.template to make a "+self._config_file+". "+err.message)
            sys.exit(1)

        config = yaml.load_all(stream, Loader=yaml.FullLoader)

        return config

    def get_value(self, key):
        '''

        :param key: see config.yaml file for keys. Keys are namespaced with '/' operator
                    for example - spark_application_config/spark.executor.cores
                    A hierarchical depth of 2 is assumed.
        :return: string value from config file
        '''
        keyns = key.split('/')

        if len(keyns) != 2:
            logger.error("config key must be namespaced!")
            sys.exit(1)

        for doc in self._get_config():
            for kk, vv in doc.items():
                if kk == keyns[0]:
                    ns = vv

        if ns is None:
            logger.error("key namespace "+keyns[0]+" not found")
            sys.exit(1)

        for kvps in ns:
            if keyns[1] == next(iter(kvps.keys())):
                value = next(iter(kvps.values()))
                break


        if value == '':
            logger.error("value for key "+key+" not found in "+self._config_file)
            sys.exit(1)

        logger.info("got config "+str(key)+"="+str(value))

        return value



# an instance that reads from the defaul config.yaml file
default_config = Config()

if __name__ == '__main__':
    print(default_config.get_value('spark_application_config/spark.executor.cores'))
