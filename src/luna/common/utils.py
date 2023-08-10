import itertools
import json
import os
import re
import subprocess
import tempfile
import time
import urllib
import warnings
from contextlib import ExitStack
from functools import wraps
from importlib import import_module
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import fire
import fsspec  # type: ignore
import pandas as pd
import requests
import yaml
from fsspec import open  # type: ignore
from loguru import logger
from omegaconf import MissingMandatoryValue, OmegaConf
from pyarrow.fs import copy_files

# Distinct types that are actually the same (effectively)
TYPE_ALIASES = {"itk_geometry": "itk_volume"}

# Sensitive cli inputs
MASK_KEYS = ["username", "user", "password", "pw"]


def _get_args_dict(fn, args, kwargs):
    args_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    return {**dict(zip(args_names, args)), **kwargs}


def validate_dask_address(addr: str) -> bool:
    """
    Return True if `addr` appears to be a valid address for a dask scheduler.

    The typical format for this will be something like 'tcp://192.168.0.37:8786',
    but there could be a hostname instead of an IP address, and maybe some other
    URL schemes are supported.  This function will be used to check whether a 
    user-defined dask scheduler address is plausible, or obviously invalid.
    """
    HOSTPORT_RE = re.compile(
        r"""^(?P<scheme> tcp):
        // (?P<host>\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3} |
           [A-Za-z][A-Za-z0-9.-]*[A-Za-z0-9] |
           [A-Za-z])
         : (?P<port>\d+)$""",
        re.VERBOSE
    )
    return bool(HOSTPORT_RE.match(addr))


def save_metadata(func):
    """This decorator saves metadata in output_url"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        metadata = get_config(_get_args_dict(func, args, kwargs))
        result = func(*args, **kwargs)
        if result is not None:
            metadata = metadata | result
            if "output_urlpath" in metadata:
                o = urlparse(str(metadata["output_urlpath"]))
                fs = fsspec.filesystem(
                    o.scheme, **metadata.get("output_storage_options", {})
                )
                with fs.open(Path(o.netloc + o.path) / "metadata.yml", "w") as f:
                    yaml.dump(metadata, f)
        return

    return wrapper


def local_cache_urlpath(
    file_key_write_mode: dict[str, str] = {}, dir_key_write_mode: dict[str, str] = {}
):
    """Decorator for caching url/paths locally"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_dict = _get_args_dict(func, args, kwargs)
            new_args_dict = args_dict.copy()

            filesystem = None
            tmp_dir_dest = []
            for key, write_mode in dir_key_write_mode.items():
                if not args_dict[key]:
                    continue
                storage_options_key = "storage_options"
                if "w" in write_mode:
                    storage_options_key = "output_storage_options"
                fs, dir = fsspec.core.url_to_fs(
                    args_dict[key], **args_dict.get(storage_options_key, {})
                )
                if fs.protocol != "file" and 'cache' not in fs.protocol:
                    new_args_dict[storage_options_key] = {"auto_mkdir": True}
                    tmp_dir = tempfile.TemporaryDirectory()
                    new_args_dict[key] = tmp_dir.name
                    tmp_dir_dest.append((tmp_dir, dir))

            result = None
            with ExitStack() as stack:
                for key, write_mode in file_key_write_mode.items():
                    if not args_dict[key]:
                        continue
                    storage_options_key = "storage_options"
                    if "w" in write_mode:
                        storage_options_key = "output_storage_options"
                    fs, path = fsspec.core.url_to_fs(
                        args_dict[key], **args_dict.get(storage_options_key, {})
                    )
                    if 'cache' not in fs.protocol:
                        simplecache_fs = fsspec.filesystem("simplecache", fs=fs)

                        of = simplecache_fs.open(path, write_mode)
                        stack.enter_context(of)
                        new_args_dict[key] = of.name

                result = func(**new_args_dict)

            for tmp_dir, dest in tmp_dir_dest:
                copy_files(tmp_dir.name, dest, destination_filesystem=filesystem)

            return result

        return wrapper

    return decorator


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper


def generate_uuid(urlpath: str, prefix, storage_options={}):
    """
    Returns hash of the file given path, preceded by the prefix.
    :param path: file path e.g. file:/path/to/file
    :param prefix: list e.g. ["SVGEOJSON","default-label"]
    :return: string uuid
    """

    fs, path = fsspec.core.url_to_fs(urlpath, **storage_options)

    rec_hash = str(fs.checksum(path))
    prefix.append(rec_hash)
    return "-".join(prefix)


def generate_uuid_binary(content, prefix):
    """
    Returns hash of the binary, preceded by the prefix.
    :param content: binary
    :param prefix: list e.g. ["FEATURE"]
    :return: string uuid
    """
    warnings.warn(
        "generate_uuid_binary() should not be used anymore, the UUIDs generated are not valid!"
    )

    content = BytesIO(content)

    uuid = "00000000"
    prefix.append(uuid)
    return "-".join(prefix)


def generate_uuid_dict(json_str, prefix):
    """
    Returns hash of the json string, preceded by the prefix.
    :param json_str: str representation of json
    :param prefix: list e.g. ["SVGEOJSON","default-label"]
    :return: v
    """
    # json_bytes = json.dumps(json_str).encode("utf-8")
    warnings.warn(
        "generate_uuid_dict() should not be used anymore, the UUIDs generated are not valid!"
    )

    uuid = "00000000"
    prefix.append(uuid)
    return "-".join(prefix)


def grouper(iterable, n):
    """Turn an iterable into an iterable of iterables

    'None' should not be a member of the input iterable as it is removed to handle the fillvalues

    Args:
        iterable (iterable): an iterable
        n (int): sie of chunks
        fillvalue

    Returns:
        iterable[iterable]
    """
    args = [iter(iterable)] * n
    return [
        [entry for entry in iterable if entry is not None]
        for iterable in itertools.zip_longest(*args, fillvalue=None)
    ]


def get_absolute_path(module_path, relative_path):
    """Given the path to a module file and the path, relative to the module file, of another file
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


def get_dataset_url():
    """Retrieve a "dataset URL" from the environment, may look like http://localhost:6077 or file:///absolute/path/to/dataset/dir"""
    dataset_url = os.environ.get("DATASET_URL", None)

    if dataset_url is None:
        logger.warning(
            "Requesting feature data be sent to dataset, however no dataset URL provided, please set env DATASET_URL!"
        )
    else:
        logger.info(f"Found dataset URL = {dataset_url}")

    return dataset_url


def post_to_dataset(input_feature_data, waystation_url, dataset_id, keys):
    """Interface feature data to a parquet dataset.

    Args:
        input_feature_data (str): path to input data
        waystation_url (str): URL of dataset root (either file or using waystation)
        dataset_id (str): Dataset name/ID
        keys (dict): corresponding segment keys
    """

    logger.info(f"Adding {input_feature_data} to {dataset_id} via {waystation_url}")

    segment_id = "-".join([v for _, v in sorted(keys.items())])

    logger.info(f"SEGMENT_ID={segment_id}")

    post_url = os.path.join(
        waystation_url, "datasets", dataset_id, "segments", segment_id
    )

    parsed_url = urllib.parse.urlparse(post_url)

    if "http" in parsed_url.scheme:
        # The cool way, using luna waystation

        logger.info(f"Posting to: {post_url}")

        res = requests.post(
            post_url,
            files={"segment_data": open(input_feature_data, "rb")},
            data={"segment_keys": json.dumps(keys)},
        )

        logger.info(f"{res}: {res.text}")

    elif "file" in parsed_url.scheme:
        # The less cool way, just using file paths

        segment_dir = Path(parsed_url.path)

        logger.info(f"Writing to: {segment_dir}")

        os.makedirs(segment_dir, exist_ok=True)

        data = pd.read_parquet(input_feature_data).reset_index()
        data = data.drop(columns="index", errors="ignore")
        data["SEGMENT_ID"] = segment_id
        re_indexors = ["SEGMENT_ID"]

        if keys is not None:
            for key, value in keys.items():
                data.loc[:, key] = value
                re_indexors.append(key)

        data = data.set_index(
            re_indexors
        ).reset_index()  # a trick to move columns to the left

        data.to_parquet(segment_dir.joinpath("data.parquet"))

    else:
        logger.warning("Unrecognized scheme: {parsed_url.scheme}, skipping!")


def get_config(cli_kwargs: dict):
    """Get the config with merged OmegaConf files

    Args:
        cli_kwargs (dict): CLI keyword arguments
    """
    configs = []  # type: List[Union[ListConfig, DictConfig]]

    cli_conf = OmegaConf.create(cli_kwargs)
    configs.append(cli_conf)

    # Get params from param file
    if cli_conf.get("local_config"):
        with open(cli_conf.local_config, "r", **cli_conf.storage_options) as f:
            local_conf = OmegaConf.load(f)
            configs.insert(0, local_conf)

    try:
        merged_conf = OmegaConf.to_container(
            OmegaConf.merge(*configs), resolve=True, throw_on_missing=True
        )
    except MissingMandatoryValue as e:
        raise fire.core.FireError(e)

    if merged_conf.get("output_urlpath"):
        o = urlparse(str(merged_conf.get("output_urlpath")))
        merged_conf["output_filesystem"] = "file"
        if o.scheme != "":
            merged_conf["output_filesystem"] = o.scheme
        if merged_conf["output_filesystem"] != "file" and not merged_conf.get(
            "output_storage_options"
        ):
            merged_conf["output_storage_options"] = merged_conf.get(
                "storage_options", {}
            )
        if merged_conf["output_filesystem"] == "file" and not merged_conf.get(
            "output_storage_options"
        ):
            merged_conf["output_storage_options"] = {"auto_mkdir": True}

    return merged_conf


def apply_csv_filter(input_paths, subset_csv=None, storage_options={}):
    """Filters a list of input_paths based on include/exclude logic given for either the full path, filename, or filestem.

    If using "include" logic, only matching entries with include=True are kept.
    If using "exclude" logic, only matching entries with exclude=True are removed.

    The origional list is returned if the given subset_csv is None or empty.

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
        subset_df = pd.read_csv(
            subset_csv, dtype={0: str}, storage_options=storage_options
        )

        match_type = subset_df.columns[0]
        filter_logic = subset_df.columns[1]

        if match_type not in ["path", "filename", "stem"]:
            raise RuntimeError("Invalid match type column")
        if filter_logic not in ["include", "exclude"]:
            raise RuntimeError("Invalid match type column")
    except Exception as exc:
        logger.error(exc)
        raise RuntimeError(
            "Invalid subset .csv passed, must be a 2-column csv with headers = [ (path|filename|stem), (include|exclude) ]"
        )

    if not len(subset_df) > 0:
        return input_paths

    logger.info(
        f"Applying csv filter, match_type={match_type}, filter_logic={filter_logic}"
    )

    input_path_df = pd.DataFrame(
        {"path": path, "filename": Path(path).name, "stem": Path(path).stem}
        for path in input_paths
    ).astype(str)

    df_matches = input_path_df.set_index(match_type).join(
        subset_df.set_index(match_type)
    )

    if filter_logic == "include":
        out = df_matches.loc[(df_matches["include"] == 1)]
    if filter_logic == "exclude":
        out = df_matches.loc[df_matches["exclude"] == 1]

    return list(out.reset_index()["path"])


def load_func(dotpath: str):
    """Load function in module from a parsed yaml string.

    Args:
        dotpath (str): module/function name written as a string (ie torchvision.models.resnet34)
    Returns:
        The inferred module itself, not the string representation
    """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)


class LunaCliCall:
    def __init__(self, cli_call, cli_client):
        self.cli_call = cli_call
        self.cli_client = cli_client
        print(" ".join(f"{x}" for x in cli_call))

    def run(self, step_name):
        """Run (execute) CLI Call given a 'step_name', add step to parent CLI Client once completed.
        Args:
            step_name (str): Name of the CLI call, determines output directory, can act as inputs to other CLI steps
        """
        if "/" in step_name:
            raise RuntimeError("Cannot name steps with path-like character /")

        output_dir = self.cli_client.get_output_dir(step_name)
        self.cli_call.append("-o")
        self.cli_call.append(output_dir)

        print(self.cli_call)

        out, err = subprocess.Popen(
            self.cli_call, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()

        print(f"{out.decode()}\n{err.decode()}")

        self.cli_client.cli_steps[step_name] = output_dir


class LunaCliClient:
    def __init__(self, base_dir, uuid):
        """Initialize Luna CLI Client with a base directory (the root working directory) and a UUID to track results.

        Args:
            base_dir (str): parent working directory
            uuid (str): some unique string for this instance
        """
        self.base_dir = Path(base_dir).expanduser()
        self.uuid = uuid
        self.cli_steps = {}

    def bootstrap(self, step_name, data_path):
        """Add data  (boostrap a root CLI call).

        Args:
            step_name (str): Name of the (boostrap) CLI call, determines output directory, can act as inputs to other CLI steps
            data_path (str): Input data path
        """
        self.cli_steps[step_name] = Path(data_path).expanduser()

    def configure(self, cli_resource, *args, **kwargs):
        """Configure a CLI step.

        Args:
            cli_resource (str): CLI Resource string like
            args (list): List of CLI arguements
            kwargs (list): List of CLI parameters
        Returns:
            LunaCliCall
        """
        cli_call = cli_resource.split(" ")
        for arg in args:
            if arg in self.cli_steps.keys():
                cli_call.append(self.cli_steps[arg])
            else:
                cli_call.append(Path(arg).expanduser())

        for key, value in kwargs.items():
            cli_call.append(f"--{key}")
            if type(value) is not bool:
                cli_call.append(f"{value}")

        return LunaCliCall(cli_call, self)

    def get_output_dir(self, step_name):
        """Get output_dir based on base_dir, uuid, and step name.

        Args:
            step_name (str): parent working directory
        Returns:
            output_dir (str)
        """
        output_dir = os.path.join(self.base_dir, self.uuid, step_name)

        return output_dir
