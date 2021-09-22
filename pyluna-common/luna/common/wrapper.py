import os
import logging
import dask
import asyncio

from dask.distributed import Client
from dask.distributed import worker_client, get_client, get_worker
from functools import partial

logger = logging.getLogger(__name__)

class JobExecutionError(Exception): pass

import dask.dataframe as dd

def service_runner(source_table, service_name, service_function, service_config):

    jobs = []

    for idx, row in source_table.iterrows():
        job = dask.delayed (service_function) (
                service_name, 
                row.output_key,
                row.dicom_path, 
                row.radiology_segmentation_path, 
                service_config) 
        jobs.append(job)
    print ("Submitted", len(jobs), "jobs!")

    dd.from_delayed(jobs).repartition(npartitions=5).to_parquet("/gpfs/mskmindhdp_emc/data_dev/TEST_DS", overwrite=True)




def service(service_tag):
    """
    Service interface wrapper

    Usage:
    @service('my_svc')
    my_job(args, kwargs, runner=None):
        runner.submit(sleep, 10)
    """

    def wrapped(func):

        def run_simple(namespace, index, *args, **kwargs):
            """
            Simple runner
            """
            # Tell us we are running
            logger.info (f"Initializing {service_tag} @ {namespace}/{index}")

            # See if we are on a dask worker
        
            # Jobs get an output directory and an ouput parquet slice
            output_dir = os.path.join(os.environ['MIND_GPFS_DIR'], "data_dev", namespace, index)

            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Setup ouput dir={output_dir}")

            # Kick off the job
            try:
                logger.info (f"Running job {func} with args: {args}, kwargs: {kwargs}")
                return_value = func (index, output_dir, *args, **kwargs )
            except Exception as exc:
                logger.exception(f"Job execution {func} @ {namespace}/{index} with args: {args}, kwargs: {kwargs} failed due to: {exc}", extra={ "namespace":namespace, "key": index})
                raise JobExecutionError(f"Job {func} @ {namespace}/{index} did not run successfully, caught: {exc}, please check input data!:\n {args}, {kwargs}")

            return return_value

        run_simple.__name__ = service_tag

        return run_simple

    return wrapped
