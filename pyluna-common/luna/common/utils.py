import os, json, yaml
import logging

from filehash import FileHash
from importlib import import_module
from io import BytesIO
from typing import Callable, List
from luna.common.CodeTimer import CodeTimer
import itertools

import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


# Distinct types that are actually the same (effectively)
TYPE_ALIASES = {
    'itk_geometry': 'itk_volume'
}

def to_sql_field(s):
    filter1 = s.replace(".","_").replace(" ","_")
    filter2 = ''.join(e for e in filter1 if e.isalnum() or e=='_')
    return filter2

def to_sql_value(s):
    if isinstance(s, str): return f"'{s}'"
    if not s==s:  return 'Null'
    if s is None: return 'Null'
    else: return f"{s}"


def clean_nested_colname(s):
    """
    Removes map name for MapType columns.
    e.g. metadata.SeriesInstanceUID -> SeriesInstanceUID
    """
    return s[s.find('.')+1:]

def generate_uuid(path, prefix):
    """
    Returns hash of the file given path, preceded by the prefix.
    :param path: file path e.g. file:/path/to/file
    :param prefix: list e.g. ["SVGEOJSON","default-label"]
    :return: string uuid
    """
    posix_file_path = path.split(':')[-1]

    rec_hash = FileHash('sha256', chunk_size=65536).hash_file(posix_file_path)
    prefix.append(rec_hash)
    return "-".join(prefix)


def rebase_schema_numeric(df):
    """
    Tries to convert all columns in a dataframe to numeric types, if possible, with integer types taking precident

    Note: this is an in-place operation
    
    Args:
        df (pd.DataFrame): dataframe to convert columns
    """
    for col in df.columns:
        if df[col].dtype != object: continue

        df[col] = df[col].astype(float, errors='ignore')
        df[col] = df[col].astype(int,   errors='ignore')


def generate_uuid_binary(content, prefix):
    """
    Returns hash of the binary, preceded by the prefix.
    :param content: binary
    :param prefix: list e.g. ["FEATURE"]
    :return: string uuid
    """
    content = BytesIO(content)

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        uuid = FileHash('sha256').hash_file(content)

    prefix.append(uuid)
    return "-".join(prefix)


def generate_uuid_dict(json_str, prefix):
    """
    Returns hash of the json string, preceded by the prefix.
    :param json_str: str representation of json
    :param prefix: list e.g. ["SVGEOJSON","default-label"]
    :return: v
    """
    json_bytes = json.dumps(json_str).encode('utf-8')

    import EnsureByteContext
    with EnsureByteContext.EnsureByteContext():
        uuid = FileHash('sha256').hash_file(BytesIO(json_bytes))

    prefix.append(uuid)
    return "-".join(prefix)

def does_not_contain(token, value):
    """
    Validate that `token` is not a substring of `value`

    :param: token: string e.g. : | .
    :param: value: dictionary, list, or str
    """
    if isinstance(value, str):
        if token in value:
            raise ValueError(f"{value} cannot contain {token}")

    if isinstance(value, list):
        if any([token in v for v in value]):
            raise ValueError(str(value) + f" cannot contain {token}")

    if isinstance(value, dict):
        if any([isinstance(key, str) and token in key or isinstance(val, str) and token in val for key,val in value.items()]):
            raise ValueError(str(value) + f" cannot contain {token}")

    return True


def replace_token(token, token_replacement, value):
    """
    Replace `token` with `token_replacement` in `value`

    :param: token: string e.g. : | .
    :param: token_replacement: string e.g. _ -
    :param: value: dictionary, list, or str
    """
    if isinstance(value, str):
        return value.replace(token, token_replacement)

    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value.append(v.replace(token, token_replacement))
        return new_value

    if isinstance(value, dict):
        new_value = {}
        for key,val in value.items():
            new_key, new_val = key, val
            if isinstance(key, str):
                new_key = key.replace(token, token_replacement)
            if isinstance(val, str):
                new_val = val.replace(token, token_replacement)
            new_value[new_key] = new_val

        return new_value

    return value



def grouper(iterable, n):
    """ Turn an iterable into an iterable of iterables

    'None' should not be a member of the input iterable as it is removed to handle the fillvalues
    
    Args:
        iterable (iterable): an iterable
        n (int): sie of chunks
        fillvalue
    
    Returns:
        iterable[iterable]
    """
    args = [iter(iterable)] * n
    return [[entry for entry in iterable if entry is not None]
            for iterable in itertools.zip_longest(*args, fillvalue=None)]


def get_method_data(cohort_id, method_id):
    """
    Return method dict

    :param: cohort_id: string
    :param: method_id: string
    """

    method_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data", cohort_id, "methods")
    with open(os.path.join(method_dir, f'{method_id}.json')) as json_file:
        method_config = json.load(json_file)['params']
    return method_config

def get_absolute_path(module_path, relative_path):
    """ Given the path to a module file and the path, relative to the module file, of another file
    that needs to be referenced in the module, this method returns the absolute path of the file
    that needs to be referenced.

    This method makes it possible to resolve absolute paths to files in any environment a
    module and the referenced files are deployed to.

    :param module_path path to the module. Use '__file__' from the module.
    :param relative_path path to the file that needs to be referenced by the module. The path must
    be relative to the module.
    :return absolute path to file with the specified relative_path
    """
    path = os.path.join(os.path.dirname(module_path), relative_path)

    # resolve any back-paths with ../ to simplify absolute path
    return os.path.realpath(path)

def validate_params(given_params: dict, params_list: List[tuple]):
    """Ensure that a dictonary of params or keyword arguments is correct given a parameter list

    Checks that neccessary parameters exist, and that their type can be casted corretly. There's special logic for list, json, and dictonary types

    Args:
        given_params (dict): keyword arguments to check types
        params_list (List[tuple]): param list, where each element is the parameter (name, type)

    Returns:
        dict: Validated and casted keyword argument dictonary
    """
    logger = logging.getLogger(__name__)

    d_params = {}
    for name, dtype in params_list:
        if given_params.get(name, None) == None: 
            raise RuntimeError(f"Param {name} of type {dtype} was never set, but required by transform, please check your input variables.")
        try:
            if 'List' in str(dtype):
                if type (given_params[name])==list:
                    d_params[name] = given_params[name]
                else:
                    d_params[name] = [dtype.__args__[0](s) for s in given_params[name].split(',')]
            elif dtype==dict:
                if type (given_params[name])==dict:
                    d_params[name] = given_params[name]
                else:
                    d_params[name] = {s.split('=')[0] : s.split('=')[1] for s in given_params[name].split(',')}
            elif dtype==json:
                if type (given_params[name])==dict:
                    d_params[name] = given_params[name]
                else:
                    d_params[name] = json.loads(given_params[name])
            elif type(dtype)==type:
                d_params[name] = dtype(given_params[name])
            elif type(dtype) == _GenericAlias: # check for parameterized generics like List[str]
                d_params[name] = given_params[name]
            else:
                raise RuntimeError(f"Type {type(dtype)} invalid!")

        except ValueError as exc:
            raise RuntimeError(f"Param {name} could not be cast to {dtype} - {exc}")

        except RuntimeError as e:
            raise e
        logger.info (f"Param {name} set = {d_params[name]}")
    return d_params


def expand_inputs(given_params: dict):
    """For special input_* parameters, see if we should infer the input given an output/result directory

    Args:
        given_params (dict): keyword arguments to check types

    Returns:
        dict: Input- expanded keyword argument dictonary
    """
    d_params = {}

    for key, value in given_params.items():
        if 'input_' in key: # We want to treat input_ params a bit differently

            # For some inputs, they may be defined as a directory, where metadata about them is at the provided directory path
            expected_metadata = os.path.join(value, 'metadata.yml')
            print (expected_metadata)
            if os.path.isdir(value) and os.path.exists(expected_metadata): # Check for this metadata file
                # We supplied an inferred input from some previous output, so figure it out from the metadata of this output fold

                with open(expected_metadata, 'r') as yaml_file:
                    metadata = yaml.safe_load(yaml_file)

                # Output names/slots are same as input names/slots, just without input_ prefix
                input_type = key.replace('input_', '')

                # Convert any known type aliases 
                input_type = TYPE_ALIASES.get(input_type, input_type)

                # Query the metadata dictionary for that type
                expanded_input = metadata.get (input_type, None)

                # Tree output_directories should never be passed to functions which cannot accept them
                if expanded_input is None:
                    raise RuntimeError (f"No matching output slot of type [{key.replace('input_', '')}] at given input directory")

                logger.info(f"Expanded input {value} -> {expanded_input}")

                d_params[key] = expanded_input
            else:
                d_params[key] = value
        else:
            d_params[key] = value

    return d_params


def cli_runner(cli_kwargs: dict, cli_params: List[tuple], cli_function: Callable[..., dict]):
    """For special input_* parameters, see if we should infer the input given an output/result directory

    Args:
        cli_kwargs (dict): keyword arguments from the CLI call
        cli_params (List[tuple]): param list, where each element is the parameter (name, type)
        cli_function (Callable[..., dict]): cli_function entry point, should accept exactly the arguments given by cli_params
    
    Returns:
        None

    """
    logger.info (f"Running {cli_function} with {cli_kwargs}")
    kwargs = {}
    if not 'output_dir' in cli_kwargs.keys():
        raise RuntimeError("CLI Runners assume an output directory")

    # Get params from param file
    if cli_kwargs.get('method_param_path'):
        with open(cli_kwargs.get('method_param_path'), 'r') as yaml_file:
            yaml_kwargs = yaml.safe_load(yaml_file)
        kwargs.update(yaml_kwargs) # Fill from json
    
    for key in list(cli_kwargs.keys()):
        if cli_kwargs[key] is None: del cli_kwargs[key]

    # Override with CLI arguments
    kwargs.update(cli_kwargs)

    kwargs = validate_params(kwargs, cli_params)

    # Expand implied inputs
    kwargs = expand_inputs(kwargs)

    output_dir = kwargs['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Nice little log break
    print("\n" + "-"*35 + f' Running transform::{cli_function.__name__} ' + "-" *35 + "\n")

    result = cli_function(**kwargs)
    
    kwargs.update(result)

    with open(os.path.join(output_dir, "metadata.yml"), "w") as fp:
        yaml.dump(kwargs, fp)
        # json.dump(kwargs, fp, indent=4, sort_keys=True)
    
    logger.info("Done.")

def apply_csv_filter(input_paths, subset_csv=None):
    """ Filteres a list of input_paths based on include/exclude logic given for either the full path, filename, or filestem

    If using "include" logic, only matching entries with include=True are kept.  
    If using "exclude" logic, only matching entries with exclude=True are removed.

    The origional list is returned if the given subset_csv is None or empty

    Args:
        input_paths (list[str]): list of input paths to filter
        subset_csv (str): path to a csv with subset/filter information/flags
    Returns
        list[str]: filtered list
    Raises:
        RuntimeError: If the given subset_csv is invalid
    """

    if not len(subset_csv) > 0 or subset_csv is None: 
        return input_paths
    if not os.path.exists(subset_csv): 
        return input_paths


    try:
        subset_df     = pd.read_csv(subset_csv, dtype={0: str})

        match_type   = subset_df.columns[0]
        filter_logic = subset_df.columns[1]

        if not match_type   in ['path', 'filename', 'stem']: raise RuntimeError("Invalid match type column")
        if not filter_logic in ['include', 'exclude']:       raise RuntimeError("Invalid match type column")
    except: 
        raise RuntimeError("Invalid subset .csv passed, must be a 2-column csv with headers = [ (path|filename|stem), (include|exclude) ]")

    if not len(subset_df) > 0: return input_paths

    logger.info(f"Applying csv filter, match_type={match_type}, filter_logic={filter_logic}")
    
    input_path_df = pd.DataFrame( {'path':path, 'filename':Path(path).name, 'stem':Path(path).stem} for path in input_paths).astype(str)

    df_matches = input_path_df.set_index(match_type).join(subset_df.set_index(match_type))

    if filter_logic=="include":
        out =  df_matches.loc[(df_matches['include'] == True)]
    if filter_logic=="exclude":
        out =  df_matches.loc[~(df_matches['exclude'] == True)]
    
    return list(out.reset_index()['path'])

def load_func(dotpath : str):
    """load function in module from a parsed yaml string

    Args:
        dotpath (str): module/function name written as a string (ie torchvision.models.resnet34)
    Returns: 
        The inferred module itself, not the string representation
    """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)
