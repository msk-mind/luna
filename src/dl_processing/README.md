# Radiology Deep Learning Preprocessing

PoC scripts to create features based on mock Scan and Annotation tables.

### Setup

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Test Dataset

Test data, along with example CSVs are on the GPFS mount.
For mock silver tables (scan, annotation), see `/gpfs/mskmind_ess/eng/radiology/dl_preprocess`
For image and segmentation data, see `/gpfs/mskmind_ess/eng/radiology/data`


### Create Mock Scan and Annotation tables in Delta format

Update the input CSV paths and delta table paths, then run:

```
python create_mock_tables.py
```


### Run preprocess job

This job
1. Reads Scan and Annotation delta tables.
2. Writes a feature table with additional columns (preprocessed_seg_path, preprocessed_img_path, preprocessed_target_spacing)
3. Resamples the image (.mhd) and segmentation (.mha) and saves the "feature files" in .npy

Make sure scan and annotation tables under {base_directory}/tables/scan and {base_directory}/tables/annotation.
Feature table and feature file location will be written under  {base_directory}/features/feature_table and {base_directory}features/feature_files.

Run:
```
python preprocess_feature.py --spark_master_uri {spark_master_uri} --base_directory {path/to/delta/table/parent/directory} --target_spacing {x_spacing} {y_spacing} {z_spacing}
```


### TODO

DONE - Take target spacing parameter, table paths as arguments (using click)
- Embed .npy in the parquet/delta tables if needed
- Using Spark UDF, foreachPartition resulted in degraded performance (~5 min) compared to using Parallel (~1 min). Investigate if there is a better way to iterate over rows in Spark.
- Performance test on Spark cluster

