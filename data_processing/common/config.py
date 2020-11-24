'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''

import yaml
import yamale
from jsonpath_ng import parse
from data_processing.common.custom_logger import init_logger

logger = init_logger()

class ConfigSet():
    '''
    This is a singleton class that can load a collection of configurations from yaml files.

    ConfigSet loads configurations from yaml files only once on first invocation of this class with
    the specified yaml file. The class then maintains the configuration in memory in a singleton instance. All new
    invocations of this class will serve up the same configuration.

    Each configuration in the collection is identified by a logical name.

    If a new invocation of this class is created with an existing logical name and a different yaml file, the singleton
    instance replaces the existing configuration with the newly specified yaml file for the given logical name.
    '''

    __CONFIG_MAP = {}  # maps logical name to yaml config file name
    __SCHEMA_MAP = {}  # maps logical name to schema file of yaml config file
    __INSTANCE = None    # singleton instance containing the collection of configs keyed by logical name


    def __new__(cls, name, config_file, schema_file=None):
        # initialize singleton
        if ConfigSet.__INSTANCE is None:
            ConfigSet.__INSTANCE = object.__new__(cls)
            ConfigSet.__INSTANCE.__config = {}

        # load or reload config into memory
        if name not in ConfigSet.__CONFIG_MAP.keys() or \
            ConfigSet.__CONFIG_MAP[name] != config_file:
            ConfigSet.__CONFIG_MAP[name] = config_file
            ConfigSet.__INSTANCE.__config[name] = ConfigSet._load_config(cls, name)

            # add schema and validate config
            if schema_file is not None:
                ConfigSet.__SCHEMA_MAP[name] = schema_file
                ConfigSet._validate_config(cls, name)

        return ConfigSet.__INSTANCE


    def __init__(self, name, config_file, schema_file=None):
        '''
        :param name logical name to be given for this configuration
        :param config_file the config file to load
        :param schema_file a schema file for the yaml configuration (optional)
        :raises yamale.yamale_error.YamaleError if config file is invalid when validated against the schema
        '''
        pass  # see __new__() method implementation

    def _validate_config(cls, name):
        config_file = ConfigSet.__CONFIG_MAP[name]
        schema_file = ConfigSet.__SCHEMA_MAP[name]
        logger.info("validating config " + config_file + " against schema " + schema_file + " for " + name)
        schema = yamale.make_schema(schema_file)
        data = yamale.make_data(config_file)
        yamale.validate(schema, data)

    def _load_config(cls, name):
        '''

        :param name: logical name of the config to load
        :return: config generator object

        :raises: IOError if yaml config file for the specified logical name cannot be found
        '''
        # read config file
        config_file = ConfigSet.__CONFIG_MAP[name]
        logger.info("loading config file "+config_file)


        try:
            stream = open(config_file, 'r')
        except IOError as err:
            logger.error("unable to find a config file with name "+config_file+
                  ". Please use config.yaml.template to make a "+config_file+". "+str(err))
            raise err
        
        config = {}
        for items in yaml.load_all(stream, Loader=yaml.FullLoader):
            config.update(items)

        return config


    def get_value(self, name, jsonpath):
        '''
        Gets the value for the specified jsonpath from the specified configuration.

        :param name: logical name of the configuration
        :param jsonpath: see config.yaml to generate a jsonpath. See https://pypi.org/project/jsonpath-ng/
                         jsonpath expressions may be tested here - https://jsonpath.com/
        :return: string value from config file
        :raises: ValueError if no match is found for the specified exception
        '''

        jsonpath_expression = parse(jsonpath)

        match = jsonpath_expression.find(ConfigSet.__INSTANCE.__config[name])

        if len(match) == 0:
            err = 'unable to find a config value for jsonpath: '+jsonpath
            logger.error(err)
            raise ValueError(err)

        return match[0].value

    def get_names(self):
        '''

        :return: a list of logical names of the configs stored in this instance.
        '''
        return list(ConfigSet.__INSTANCE.__config.keys())


if __name__ == '__main__':
    c1 = ConfigSet('app_config', 'tests/data_processing/common/test_config.yml')
    c2 = ConfigSet('app_config', 'tests/data_processing/common/test_config.yml')
    c3 = ConfigSet('app_config', 'tests/data_processing/common/test_config.yml')

    print(str(c1) + ' ' + str(c1.get_value('app_config', '$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c2) + ' ' + str(c2.get_value('app_config', '$.spark_application_config[:1]["spark.executor.cores"]')))
    try:
        print(str(c3) + ' ' + str(c3.get_value('app_config', '$.spark_application_config[:1]["doesnt_exist"]')))
    except ValueError as ve:
        print("got expected value error: "+str(ve))

    c4 = ConfigSet(name='data_config',
                   config_file="tests/data_processing/common/test_data_ingestion_template.yml",
                   schema_file="data_ingestion_template_schema.yml")

    print(str(c4) + ' ' + str(c4.get_value('data_config', '$.REQUESTOR')))