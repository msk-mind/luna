from data_processing.common.Container import Container
from data_processing.common.Container import Node
from data_processing.common.config import ConfigSet

from data_processing.radiology.cli.collect_csv_segment import collect_result_segment_with_container
from data_processing.radiology.cli.extract_radiomics   import extract_radiomics_with_container

import concurrent.futures

executor = concurrent.futures.ProcessPoolExecutor(20)

cfg = ConfigSet("APP_CFG", "config.yaml")

container = Container(cfg)
container.createNamespace("my-analysis")

for patient in ['QIN-BREAST-01-0001', 'QIN-BREAST-01-0002']:
    container = Container(cfg).setNamespace   ("my-analysis")
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

    radiomics_job_config = {
            "image_input_tag": "main_scan",
            "label_input_tag": "annotation_ata",
            "job_tag" : "my_radiomics",
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
    
    collect_job_config = {
            "input_tags": ["my_radiomics"],
            "output_container": "my_results_v2",
    }

    pipeline = [
        (extract_radiomics_with_container,      radiomics_job_config),
        (collect_result_segment_with_container, collect_job_config),
    ]

    # container.runLocal(pipeline)
    container.runProcessPoolExecutor(pipeline, executor)

executor.shutdown()

