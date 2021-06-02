import os
import logging
import dask
import asyncio

from dask.distributed import Client
from dask.distributed import worker_client, get_client, get_worker
from functools import partial

from distributed.threadpoolexecutor import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def get_or_create_dask_client():
    try:
        client = get_client()
    except ValueError:
        client = Client(threads_per_worker=1)
    return client

def get_local_dask_directory():
    local_directory = dask.config.get("temporary-directory") or os.getcwd()
    local_directory = os.path.join(local_directory, "dask-worker-space")
    return local_directory


def dask_worker_runner(func):
    """
    This methods decorates functions run on dask workers with an async function call
    Namely, this allows us to manage the execution of a function a bit better, and especially, to exit job execution if things take to long (1hr)

    Here, the function func is run in a background thread, and has access to the dask schedular through the 'runner'.
    Critically, sumbission to this runner/client looks the same regardless of if it occurs in a sub-process/thread

    Mostly, this is a workaround to impliment some form of timeout when running very long-tasks on dask. 
    While one cannot (or should not) kill the running thread, Dask will cleanup the child tasks eventually once all jobs finish.

    Usage:
    @dask_worker_runner
    my_job(args, kwargs, runner=None):
        runner.submit(sleep, 10)
    """

    async def wrapped(*args, **kwargs):
        # We'll run our function in a background thread
        executor = ThreadPoolExecutor(max_workers=1)

        loop = asyncio.get_event_loop()        
        
        # Get our current dask worker, functions wrapped with this method can only be run on dask workers
        logger.info ("Initializing job... getting parent worker")
        try:
            worker = get_worker()
        except ValueError as exc:
            logger.error("Data-processing job called without parent dask worker")
            raise RuntimeError("Data-processing job called without parent dask worker")
        except Exception as exc:
            logger.exception(f"Unknown exception when getting dask worker")

        logger.info (f"Successfully found worker {worker}")
        logger.info (f"Running job {func} with args: {args}, kwargs: {kwargs}")

        # Get our worker client, and pass as a dask client exector
        with worker_client() as runner:
            kwargs['runner'] = runner
            job = loop.run_in_executor(executor, partial(func, *args, **kwargs))
        
        # Move on from job if things take more than hour
        done, pending = await asyncio.wait([job], timeout=3600)

        # Do some cleanup
        if len(pending) != 0:
            logger.warning ("Killing pending tasks!")
            for task in pending: task.cancel()

        executor.shutdown(wait=False)

        # Get the return value
        if len(done) == 1:
            return_value = done.pop().result()
        else:
            return_value = None

        logger.info (f"Done running job, returning {return_value}")

        return return_value
    
    def run(*args, **kwargs):
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(wrapped(*args, **kwargs))
          
    return run

