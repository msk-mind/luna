import sys

from data_processing.common.DataStore import DataStore
from data_processing.common.DataStore import Node
from data_processing.common.config import ConfigSet

from data_processing.radiology.cli.randomize_contours  import randomize_contours_with_container
from data_processing.radiology.cli.extract_radiomics   import extract_radiomics_with_container
from data_processing.radiology.cli.collect_csv_segment import collect_result_segment_with_container


if __name__=='__main__':
    backend = sys.argv[1]
    print ("Submitting to", backend)

    if backend=='pool':
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(20)

    elif backend=='dask':
        from dask.distributed import Client
        dask_client = Client()
        
        # This souldn't be neccessary once we have stable releases of data_processing
        dask_client.upload_file('data_processing/radiology/cli/collect_csv_segment.py')
        dask_client.upload_file('data_processing/radiology/cli/extract_radiomics.py')
        dask_client.upload_file('data_processing/radiology/cli/randomize_contours.py')
        dask_client.upload_file('data_processing/common/DataStore.py')
        dask_client.upload_file('data_processing/common/Node.py')


    cfg = ConfigSet("APP_CFG", "config.yaml")

    container = DataStore(cfg)
    container.createNamespace("my-analysis")

    futures = []

    for patient in ['QIN-BREAST-01-0001', 'QIN-BREAST-01-0002']:
        container = DataStore(cfg).setNamespace("my-analysis")

        container.createContainer(patient, 'patient')
        container.setContainer   (patient)

        image = Node("VolumetricImage", "main_scan")
        image.set_data(f"qin-test-data/{patient}.mhd")
        image.set_aux (f"qin-test-data/{patient}.raw")
        container.add(image)

        label = Node("VolumetricLabel", "annotation_ata")
        label.set_data(f"qin-test-data/{patient}.mha")
        container.add(label)

        container.saveAll()

        pertubation_job_config = {
            "job_tag": "mirp_pertubation",
            "image_input_tag": "main_scan",
            "label_input_tag": "annotation_ata"
        }

        radiomics_job_config_original = {
                "image_input_tag": "main_scan",
                "label_input_tag": "annotation_ata",
                "job_tag" : "my_radiomics_original",
                "strictGeometry": True,
                "enableAllImageTypes": True,
                "RadiomicsFeatureExtractor": {
                    "interpolator": "sitkBSpline",
                    "resampledPixelSpacing": [1.25, 1.25, 1.25],
                    "binWidth": 10,
                    "verbose": "True",
                    "label":1,
                    "geometryTolerance":1e-08
            }
        }

        radiomics_job_config_pertubation = {
                "image_input_tag": "mirp_pertubation",
                "label_input_tag": "mirp_pertubation",
                "job_tag" : "my_radiomics_pertubation",
                "strictGeometry": True,
                "enableAllImageTypes": True,
                "usingPertubations": True,
                "RadiomicsFeatureExtractor": {
                    "interpolator": "sitkBSpline",
                    "resampledPixelSpacing": [1.25, 1.25, 1.25],
                    "binWidth": 10,
                    "verbose": "True",
                    "label":1,
                    "geometryTolerance":1e-08
            }
        }
        
        collect_job_config = {
                "input_tags": ["my_radiomics_original", "my_radiomics_pertubation"],
                "output_container": "my_results",
        }

        pipeline = [
            (randomize_contours_with_container,     pertubation_job_config),
            (extract_radiomics_with_container,      radiomics_job_config_original),
            (extract_radiomics_with_container,      radiomics_job_config_pertubation),
            (collect_result_segment_with_container, collect_job_config),
        ]

        if backend=='local':
            container.runLocal(pipeline)
        elif backend=='pool':
            futures.append ( container.runProcessPoolExecutor(pipeline, executor) )
        elif backend=='dask':
            futures.append ( container.runDaskDistributed(pipeline, dask_client)  )

    # Block in this process until everything completes
    for future in futures:
        print (future.result())