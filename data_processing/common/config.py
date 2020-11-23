'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''
import sys
import os

# sys.path.append(os.path.abspath('../'))

import yaml
import yamale
import sys
from jsonpath_ng import parse
from data_processing.common.custom_logger import init_logger

logger = init_logger()

class Config():
    '''
    This class loads the configuration from a config yaml file only once on first invocation of this class with
    the specified config file. The class then maintains the configuration in memory in a singleton instance. All new
    instances of this class that are created for the same config file specified as the argument, will use the same
    singleton instance.

    If a new instance of this class is created with a new config file specified as the argument, the singleton instance
    is recreated and the class loads the config from the new config file.
    '''

    __CONFIG_FILE = 'config.yaml'  # config file 
    __INSTANCE = None              # singleton instance


    def __new__(cls, config_file, schema_file=None):
        if Config.__INSTANCE is None or Config.__CONFIG_FILE != config_file:
            Config.__CONFIG_FILE = config_file

            if schema_file is not None:
                Config.__SCHEMA_FILE = schema_file
                Config._validate_config(cls)

            Config.__INSTANCE = object.__new__(cls)
            Config.__INSTANCE.__config = Config._load_config(cls)
        return Config.__INSTANCE


    def __init__(self, config_file, schema_file=None):
        ''':param config_file the config file to load. If none is provided, the class defaults to 'config.yaml' in
        the base directory'''
        pass

    def _validate_config(cls):
        config_file = Config.__CONFIG_FILE
        schema_file = Config.__SCHEMA_FILE
        logger.info("validating config " + config_file + " against schema " + schema_file)
        schema = yamale.make_schema(schema_file)
        data = yamale.make_data(config_file)
        yamale.validate(schema, data)

    def _load_config(cls):
        '''

        :param config_file: Default config file is config.yaml
        :return: config generator object
        '''
        # read config file
        config_file = Config.__CONFIG_FILE
        logger.info("loading config file "+config_file)


        try:
            stream = open(config_file, 'r')
        except IOError as err:
            logger.error("unable to find a config file with name "+config_file+
                  ". Please use config.yaml.template to make a "+config_file+". "+err.message)
            sys.exit(1)
        
        configs = {}
        for items in yaml.load_all(stream, Loader=yaml.FullLoader):
            configs.update(items)

        return configs


    def get_value(self, jsonpath):
        '''
        Gets the value from the config file for the specified jsonpath.

        :param jsonpath: see config.yaml to generate a jsonpath. See https://pypi.org/project/jsonpath-ng/
                         jsonpath expressions may be tested here - https://jsonpath.com/
        :return: string value from config file
        '''

        jsonpath_expression = parse(jsonpath)

        match = jsonpath_expression.find(Config.__INSTANCE.__config)

        if len(match) == 0:
            logger.error('unable to find a config value for jsonpath: '+jsonpath)
            return None

        return match[0].value



if __name__ == '__main__':
    c1 = Config('config.yaml')
    c2 = Config('config.yaml')
    c3 = Config('config.yaml')

    print(str(c1) + ' ' + str(c1.get_value('$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c2) + ' ' + str(c2.get_value('$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c3) + ' ' + str(c3.get_value('$.spark_application_config[:1]["doesnt_exist"]')))

    c4 = Config(config_file="data_ingestion_template.yaml", schema_file="data_ingestion_template_schema.yml")

    print(str(c4) + ' ' + str(c4.get_value('$.requestor')))
