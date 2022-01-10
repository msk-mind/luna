import os
import pandas as pd

import pyarrow.parquet as pq
import pyarrow as pa

import logging

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

from hashlib import md5

from luna.common.utils import validate_params

def _call_impl(self, *args, **kwargs):
    r"""Wrapping call function.
    """
    execute_call =  self.execute

    root_workdir = self.root_workdir

    transform_name = self.transform_name

    os.makedirs(root_workdir, exist_ok=True)

    input = args
    if kwargs:
        self.logger.warning (f"Using keyward arguments {kwargs} can effect function caching.")
        input = args + (kwargs,)
    
    config_hash = md5((transform_name + str([getattr(self, param) for param in self.__params__])).encode()).hexdigest()
    input_hash  = md5((str(input)).encode()).hexdigest()
    self.logger.debug (f"Function hash={config_hash}, Input hash = {input_hash}")

    self.workdir = os.path.join(root_workdir, transform_name, config_hash, input_hash)
    os.makedirs(self.workdir, exist_ok=True)
    self.logger.debug (f"Allocated working directory: {self.workdir}")

    if os.path.exists(os.path.join(self.workdir, "FAILURE")):
        self.logger.debug ("Tried, but failed")
    elif os.path.exists(os.path.join(self.workdir, "SUCCESS")):
        print ("Cached result")

    result = None

    try:
        result = execute_call(*args, **kwargs)
    except Exception as e:
        self.logger.warning ("Execute call failed: " + str(e))
        with open(os.path.join(self.workdir, "FAILURE"), 'w') as f: pass
    else:
        self.logger.warning ("Execute call succeeded")
        with open(os.path.join(self.workdir, "SUCCESS"), 'w') as f: pass

    return result
    

def _execute_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


class BaseTransform:
    
    def __init__(self):
        self.transform_name = type(self).__name__
        self.logger = logging.getLogger(self.transform_name)
        self.root_workdir = os.environ.get('LUNA_WORKDIR', None)

        if self.root_workdir is None: raise RuntimeError("LUNA_WORKDIR was not set, stopping.")
    
    __call__ : Callable[..., Any] = _call_impl
        
    execute: Callable[..., Any] = _execute_unimplemented


class TestTransform(BaseTransform):
    __params__ = ['prefix']
    def __init__(self, prefix):
        super(TestTransform, self).__init__()
        self.prefix = prefix
    def execute(self, r1):
        self.logger.info("writing " + self.prefix + r1)

        if not r1[0] == 'j': raise RuntimeError("Can only write j names!")

        with open(self.workdir + "/result.txt", 'w') as f:
            f.write(self.prefix + r1)




class SaveParquet:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
    def __call__(self, input_data, segment_id, extra_metadata=None):

        os.makedirs(self.dataset_dir, exist_ok=True)

        df = pd.read_csv(input_data)

        for col in df.columns:
            df[col] = df[col].astype(float, errors='ignore')
        
        df['segment_id'] = segment_id
        df = df.set_index('segment_id')

        for key, val in extra_metadata.items():
            df[key] = val

        output_filename = os.path.join(self.dataset_dir, f"{segment_id}.parquet")

        pq.write_table(pa.Table.from_pandas(df), output_filename)

        return output_filename

