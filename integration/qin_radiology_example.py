import sys, logging

from luna.common.DataStore import DataStore
from luna.common.DataStore import Node
from luna.common.config import ConfigSet
from luna.common.CodeTimer import CodeTimer
from luna.common.PipelineBuilder import load


if __name__=='__main__':
    backend = sys.argv[1]
    with CodeTimer(logging.getLogger(), f"Running qin_tcia_example with backend=[{backend}]"):

        if backend=='pool':
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(20)

        elif backend=='dask':
            from dask.distributed import Client
            dask_client = Client(processes = True, threads_per_worker=1, n_workers=10)
            
            # This souldn't be neccessary once we have stable releases of luna
            dask_client.upload_file('luna/radiology/cli/collect_csv_segment.py')
            dask_client.upload_file('luna/radiology/cli/extract_radiomics.py')
            dask_client.upload_file('luna/radiology/cli/randomize_contours.py')
            dask_client.upload_file('luna/common/DataStore.py')
            dask_client.upload_file('luna/common/Node.py')


        cfg = ConfigSet("APP_CFG", "config.yaml")

        container = DataStore(cfg)
        container.createNamespace("my-analysis")

        futures = []

        pipeline = load("luna/pipelines/qin_tcia_example.yaml")

        for patient in ['QIN-BREAST-01-0001', 'QIN-BREAST-01-0002']:
            container = DataStore(cfg).setNamespace("my-analysis")

            container.createDatastore(patient, 'patient')
            container.setDatastore   (patient)

            image = Node("VolumetricImage", "main_scan")
            image.set_data(f"qin-test-data/{patient}.mhd")
            image.set_aux (f"qin-test-data/{patient}.raw")
            container.put(image)

            label = Node("VolumetricLabel", "annotation_ata")
            label.set_data(f"qin-test-data/{patient}.mha")
            container.put(label)

            

            if backend=='local':
                container.runLocal(pipeline)
            elif backend=='pool':
                futures.append ( container.runProcessPoolExecutor(pipeline, executor) )
            elif backend=='dask':
                futures.append ( container.runDaskDistributed(pipeline, dask_client)  )

        # Block in this process until everything completes
        for future in futures:
            print (future.result())
