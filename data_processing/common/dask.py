
import logging

from dask.distributed import Client
from dask.distributed import worker_client, get_client, get_worker

logger = logging.getLogger(__name__)

def get_or_create_client():
    try:
        client = get_client()
    except ValueError:
        client = Client(threads_per_worker=1)
    return client

def job_runner(func):
    def execute_function_as_job(*args, **kwargs):
        logger.info ("Initializing job...getting parent worker")
        try:
            worker = get_worker()
        except ValueError as exc:
            raise RuntimeError("Data processing job called without parent dask worker")

        logger.info (f"Found worker {worker}")
        
        with worker_client() as runner:
            kwargs['runner'] = runner
            return_value = func(*args, **kwargs)

        logger.info ("Done running job, returning {return_value}")

        return return_value
          
    return execute_function_as_job

