import os
import logging
import dask
import asyncio

from dask.distributed import Client
from dask.distributed import worker_client, get_client, get_worker
from functools import partial

logger = logging.getLogger(__name__)

LENGTH_MANY_TASKS = 50000 # When should we warn the user they are possibly paying a high price?

def prune_empty_delayed(tasks):
    """
    A less-than-ideal method to prune empty tasks from dask tasks 
    Here we're trading CPU and time for memory.
    
    Args:
        tasks (list): list of delayed dask tasks
    
    Returns:
        list[dask.delayed]: a reduced list of delayed dask tasks
    """
    @dask.delayed
    def mock_return(task):
        return task

    if len(tasks) >= LENGTH_MANY_TASKS:
        logger.warning(f"Hope you're okay with a length={len(tasks)} for loop!")

    return [mock_return(task) for task in dask.compute(*tasks) if task is not None]

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

class JobExecutionError(Exception): pass

def with_event_loop(func):
    """
    This method decorates functions run on dask workers with an async function call
    Namely, this allows us to manage the execution of a function a bit better, and especially, to exit job execution if things take too long (1hr)

    Here, the function func is run in a background thread, and has access to the dask schedular through the 'runner'.
    Critically, sumbission to this runner/client looks the same regardless of if it occurs in a sub-process/thread

    Mostly, this is a workaround to impliment some form of timeout when running very long-tasks on dask. 
    While one cannot (or should not) kill the running thread, Dask will cleanup the child tasks eventually once all jobs finish.

    Examples:
        >>> @with_dask_event_loop
        >>> my_job(args, kwargs, runner=None):
        >>>     runner.submit(sleep, 10)
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
    
def dask_job(job_name):
    """
    The simplier version of a dask job decorator, which only provides the worker_client as a runner to the calling function

    Examples:
        >>> @dask_job('my_job')
        >>> my_job(args, kwargs, runner=None):
        >>>     runner.submit(sleep, 10)
    """

    def wrapped(func):

        def run_simple(namespace, index, *args, **kwargs):
            """
            Only provides runner object to method, no threading
            """
            # Tell us we are running
            logger.info (f"Initializing {job_name} @ {namespace}/{index}")

            # See if we are on a dask worker
            try:
                worker = get_worker()
            except ValueError as exc:
                logger.warning("Could not get dask worker!")
                worker = None
            except Exception as exc:
                logger.exception(f"Unknown exception when getting dask worker")
                worker = None

            logger.info (f"Successfully found worker {worker}")

            # Jobs get an output directory and an ouput parquet slice
            output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, index)
            output_ds  = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, job_name.upper() + "_DS")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_ds,  exist_ok=True)
            output_segment = os.path.join(output_ds, f"ResultSegment-{index}.parquet")

            logger.info(f"Setup ouput dir={output_dir} slice={output_ds}")

            # Kick off the job
            try:
                logger.info (f"Running job {func} with args: {args}, kwargs: {kwargs}")
                return_value = func (index, output_dir, output_segment, *args, **kwargs )
            except Exception as exc:
                logger.exception(f"Job execution failed due to: {exc}", extra={ "namespace":namespace, "key": index})
                raise JobExecutionError(f"Job {func} did not run successfully, please check input data! {args}, {kwargs}")

            return return_value

        run_simple.__name__ = job_name

        return run_simple

    return wrapped
