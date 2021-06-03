import os
import logging
import dask
import asyncio

from dask.distributed import Client
from dask.distributed import worker_client, get_client, get_worker
from functools import partial

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


def with_dask_event_loop(func):
    """
    This method decorates functions run on dask workers with an async function call
    Namely, this allows us to manage the execution of a function a bit better, and especially, to exit job execution if things take too long (1hr)

    Here, the function func is run in a background thread, and has access to the dask schedular through the 'runner'.
    Critically, sumbission to this runner/client looks the same regardless of if it occurs in a sub-process/thread

    Mostly, this is a workaround to impliment some form of timeout when running very long-tasks on dask. 
    While one cannot (or should not) kill the running thread, Dask will cleanup the child tasks eventually once all jobs finish.

    Usage:
    @dask_event_loop
    my_job(args, kwargs, runner=None):
        runner.submit(sleep, 10)
    """

    async def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()        
        
        # Get our current dask worker, functions wrapped with this method can only be run on dask workers
        logger.info ("Initializing job... getting parent worker")
        try:
            worker = get_worker()
        except ValueError as exc:
            logger.error("Could not get dask worker!")
            raise RuntimeError("Data-processing job called without parent dask worker")
        except Exception as exc:
            logger.exception(f"Unknown exception when getting dask worker")

        logger.info (f"Successfully found worker {worker}")
        logger.info (f"Running job {func} with args: {args}, kwargs: {kwargs}")

        # Get our worker client, and pass as a dask client exector
        with worker_client() as runner:

            # We'll run our function in a background thread
            # executor = ProcessPoolExecutor(max_workers=1)

            # Add our runner to kwargs
            kwargs['runner'] = runner

            # Kick off the job
            job = loop.run_in_executor(worker.executor, partial(func, *args, **kwargs))
        
            # Move on from job if things take more than hour
            done, pending = await asyncio.wait([job], timeout=3600)

            # Do some cleanup
            if len(pending) != 0:
                logger.warning ("Killing pending tasks!")
                for task in pending: task.cancel()

            # executor.shutdown(wait=False)

            # Get the return value
            if len(done) == 1:
                return_value = done.pop().result()
            else:
                return_value = None

            # Logg that we're done!
            logger.info (f"Done running job, returning {return_value}")

        return return_value
    
    def run_loop(*args, **kwargs):
        """
        Uses async and threading capabilities

        Use of background thread causes this error on shutdown: 
            ERROR - asyncio - task: <Task pending coro=<HTTP1ServerConnection._server_request_loop() running at /gpfs/mskmindhdp_emc/sw/env/lib64/python3.6/site-packages/tornado/http1connection.py:817> wait_for=<Future pending cb=[<TaskWakeupMethWrapper object at 0x7f52e8259318>()]> cb=[IOLoop.add_future.<locals>.<lambda>() at /gpfs/mskmindhdp_emc/sw/env/lib64/python3.6/site-packages/tornado/ioloop.py:690]>
            Seems like some async task gets hung up in the child thread...
        """

        loop   = asyncio.new_event_loop()
        result = loop.run_until_complete(wrapped(*args, **kwargs))
        loop.close()
        return result
    
def with_dask_runner(func):

    def run_simple(*args, **kwargs):
        """
        Only provides runner object to method, no threading
        """

        logger.info ("Initializing job... getting parent worker")

        try:
            worker = get_worker()
        except ValueError as exc:
            logger.error("Could not get dask worker!")
            raise RuntimeError("Data-processing job called without parent dask worker")
        except Exception as exc:
            logger.exception(f"Unknown exception when getting dask worker")

        logger.info (f"Successfully found worker {worker}")
        logger.info (f"Running job {func} with args: {args}, kwargs: {kwargs}")

        with worker_client() as runner:

            # Add our runner to kwargs
            kwargs['runner'] = runner

            # Kick off the job
            return_value = func ( *args, **kwargs )

        return return_value
        
    return run_simple
