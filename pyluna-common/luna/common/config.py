'''
Created on October 17, 2019

@author: pashaa@mskcc.org
'''

import yaml
import yamale
import logging
from jsonpath_ng import parse
from luna.common.utils import get_absolute_path

logger = logging.getLogger(__name__)

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


    def __new__(cls, name=None, config_file=None, schema_file=None):
        # assume one or more collections have already been loaded
        if name is None or config_file is None:
            return ConfigSet.__INSTANCE

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


    def __init__(self, name=None, config_file=None, schema_file=None):
        '''
        :param name logical name to be given for this configuration. This argument only needs to be provided on first
                    invocation (optional).
        :param config_file the config file to load. This argument only needs to be provided on first
                           invocation (optional).
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

    def _parse_path(self, path):
        path_segments = path.split('::', 1)  # split just once

        if len(path_segments) != 2:
            err = 'Illegal config path: '+path+'. must be of form "name::jsonpath" ' \
                        'where name is the logical name of the configuration and jsonpath is the ' \
                                                      'jsonpath into the yaml configuration'
            logger.error(err)
            raise ValueError(err)

        return {'name': path_segments[0], 'jsonpath': path_segments[1]}


    def _get_match(self, name, jsonpath):
        jsonpath_expression = parse(jsonpath)

        return jsonpath_expression.find(ConfigSet.__INSTANCE.__config[name])


    def has_value(self, path):
        '''
        Args:
            path (str): path to a value in a configuration. The path must be of the form
            "name::jsonpath" where name is the logical name of the configuration and jsonpath is the jsonpath to value.
            see config.yaml to generate a jsonpath. See https://pypi.org/project/jsonpath-ng/ jsonpath expressions
            may be tested here - https://jsonpath.com/

        Returns:
            boolean: true if value is not an empty string, else false.

        Raises:
            ValueError: if a configuration with the specified name was never loaded

        '''
        parsed = self._parse_path(path)
        name= parsed['name']
        jsonpath = parsed['jsonpath']

        if ConfigSet.__INSTANCE is None or name not in ConfigSet.__INSTANCE.__config.keys():
            raise ValueError('configuration with logical name '+name+' was never loaded')

        if len(self._get_match(name, jsonpath)) == 0:
            return False
        else:
            return True


    def get_value(self, path):
        '''
        Gets the value for the specified jsonpath from the specified configuration.

        Args:
            path (str): path to a value in a configuration. The path must be of the form "name::jsonpath"
            where name is the logical name of the configuration and jsonpath is the jsonpath to value.
            see config.yaml to generate a jsonpath. See https://pypi.org/project/jsonpath-ng/
            jsonpath expressions may be tested here - https://jsonpath.com/

        Returns:
            str: value from config file

        Raises:
            ValueError: if no match is found for the specified exception or a configuration with
            the specified name was never loaded

        '''
        parsed = self._parse_path(path)
        name = parsed['name']
        jsonpath = parsed['jsonpath']

        if ConfigSet.__INSTANCE is None or name not in ConfigSet.__INSTANCE.__config.keys():
            raise ValueError('configuration with logical name '+name+' was never loaded')

        match = self._get_match(name, jsonpath)

        if len(match) == 0:
            err = 'unable to find a config value for jsonpath: '+jsonpath
            logger.error(err)
            raise ValueError(err)

        return match[0].value


    def get_names(self):
        '''

        :return: a list of logical names of the configs stored in this instance.
        '''
        if ConfigSet.__INSTANCE is not None:
            return list(ConfigSet.__INSTANCE.__config.keys())
        else:
            return []


    def get_keys(self, name):
        '''

        :param name: logical name of the configuration
        :return: a list of top-level keys in the config stored in this instance.
        :raises: ValueError if a configuration with the specified name was never loaded
        '''
        if ConfigSet.__INSTANCE is None or name not in ConfigSet.__INSTANCE.__config.keys():
            raise ValueError('configuration with logical name '+name+' was never loaded')

        return list(ConfigSet.__INSTANCE.__config[name].keys())

    def get_config_set(self, name):
        '''

        :param name: logical name of the configuration
        :return: a dictonary of top-level keys in the config stored in this instance.
        :raises: ValueError if a configuration with the specified name was never loaded

        '''
        if ConfigSet.__INSTANCE is None or name not in ConfigSet.__INSTANCE.__config.keys():
            raise ValueError('configuration with logical name '+name+' was never loaded')
        return ConfigSet.__INSTANCE.__config[name]



    def clear(self):
        '''
        clear the entire collection of configurations
        '''
        ConfigSet.__CONFIG_MAP = {}
        ConfigSet.__SCHEMA_MAP = {}
        ConfigSet.__INSTANCE = None


if __name__ == '__main__':
    c1 = ConfigSet('app_config', 'pyluna-pathology/tests/luna/common/test_config.yml')
    c2 = ConfigSet('app_config', 'pyluna-pathology/tests/luna/common/test_config.yml')
    c3 = ConfigSet('app_config', 'pyluna-pathology/tests/luna/common/test_config.yml')

    print(str(c1) + ' ' + str(c1.get_value('app_config::$.spark_application_config[:1]["spark.executor.cores"]')))
    print(str(c2) + ' ' + str(c2.get_value('app_config::$.spark_application_config[:1]["spark.executor.cores"]')))
    try:
        print(str(c3) + ' ' + str(c3.get_value('app_config::$.spark_application_config[:1]["doesnt_exist"]')))
    except ValueError as ve:
        print("got expected value error: "+str(ve))

    schema_file = get_absolute_path(__file__, '../data_ingestion_template_schema.yml')

    c4 = ConfigSet(name='data_config',
                   config_file="pyluna-pathology/tests/luna/common/test_data_ingestion_template.yml",
                   schema_file=schema_file)

    print(str(c4) + ' ' + str(c4.get_value('data_config::$.REQUESTOR')))
