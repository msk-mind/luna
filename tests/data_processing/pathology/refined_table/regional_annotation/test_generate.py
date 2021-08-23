import pytest
import os, shutil
from click.testing import CliRunner
import pandas,json

from data_processing.common.config import ConfigSet
from data_processing.common.sparksession import SparkConfig
from data_processing.pathology.refined_table.regional_annotation.generate import cli
import data_processing.common.constants as const
# from data_processing.pathology.common.build_geojson import build_geojson_from_annotation

from pyspark.sql.functions import  to_json, lit, collect_list, udf

project_path = "tests/data_processing/pathology/testdata/data/test-project"
geojson_table_path = project_path + "/tables/REGIONAL_GEOJSON_dsn"
geojson_app_config_path = project_path +  "/configs/REGIONAL_GEOJSON_dsn/app_config.yaml"
geojson_data_config_path = project_path + "/configs/REGIONAL_GEOJSON_dsn/data_config.yaml"

concat_geojson_table_path = project_path +  "/tables/REGIONAL_CONCAT_GEOJSON_ds"
concat_geojson_app_config_path = project_path +  "/configs/REGIONAL_CONCAT_GEOJSON_ds/app_config.yaml"
concat_geojson_data_config_path = project_path + "/configs/REGIONAL_CONCAT_GEOJSON_ds/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='tests/test_config.yaml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-pathology-refined-annot')

    yield spark

    print('------teardown------')
    clean_up_paths = [geojson_table_path, concat_geojson_table_path]
    for path in clean_up_paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        
def test_cli_geojson(spark):

    runner = CliRunner()
    result = runner.invoke(cli, 
        ['-d', 'tests/data_processing/pathology/refined_table/regional_annotation/geojson_data.yaml',
         '-a', 'tests/test_config.yaml',
         '-p', 'geojson'])

    assert result.exit_code == 0

    assert os.path.exists(geojson_app_config_path)
    assert os.path.exists(geojson_data_config_path)

    df = spark.read.format("delta").load(geojson_table_path)
    df.show(10, False)
    assert df.count() == 2
    df.unpersist()


def test_cli_concat(spark):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ['-d', 'tests/data_processing/pathology/refined_table/regional_annotation/geojson_concat_data.yaml',
                            '-a', 'tests/test_config.yaml',
                            '-p', 'concat'])

    assert result.exit_code == 0

    assert os.path.exists(concat_geojson_app_config_path)
    assert os.path.exists(concat_geojson_data_config_path)

    df = spark.read.format("delta").load(concat_geojson_table_path)
    df.show(10, False)
    assert df.count() == 1
    df.unpersist()


def test_hole_geojson(monkeypatch, spark):

    monkeypatch.setenv("MIND_GPFS_DIR", "")
    monkeypatch.setenv("HDFS_URI", "")

    # import setup
    spark.sparkContext.addPyFile("./data_processing/pathology/common/build_geojson.py")
    from build_geojson import build_geojson_from_annotation
    from build_geojson import concatenate_regional_geojsons
    from data_processing.pathology.refined_table.regional_annotation.generate import geojson_struct


    data = {'slideviewer_path': ['TEST-1.svs'],
            'slide_id': ['TEST-1'],
            'sv_project_id': [155],
            'npy_filepath': ['tests/data_processing/pathology/refined_table/regional_annotation/test_data/small_holed_annotation.npy'],
            'user': ['someuser'],
            'bmp_record_uuid': ['SVBMP-1234567'],
            'labelset': ['TEST_LABELSET'],
            'label_config': [
                str({'TEST_LABELSET': {1: 'SOME_ANNOTATION_LABEL'}})
                ],
            'contour_level': [0.5],
            'polygon_tolerance': [1],
            'geojson': [""]
            }

    df = pandas.DataFrame(data=data)
    df = spark.createDataFrame(df)

 
    df = df.groupby(["bmp_record_uuid", "labelset"]).applyInPandas(build_geojson_from_annotation, schema = df.schema)
    df.show()

    ## build concat table from geojsons table
    concatgeojson_df = df \
        .select("sv_project_id", "slideviewer_path", "slide_id", "labelset", "geojson") \
        .groupby(["sv_project_id", "slideviewer_path", "slide_id", "labelset"]) \
        .agg(collect_list("geojson").alias("geojson_list"))


    # build struct object from list of concatenated geojson (length 1)
    concatenate_regional_geojsons_udf = udf(concatenate_regional_geojsons, geojson_struct)
    concatgeojson_df = concatgeojson_df.withColumn("concat_geojson", concatenate_regional_geojsons_udf("geojson_list")).cache()
    
    # test querying concat json struct
    row = concatgeojson_df.where(f"slide_id='{data['slide_id'][0]}' and labelset='{data['labelset'][0]}'")
    geojson = row.select(to_json("concat_geojson").alias("val")).head()['val']
    assert (geojson != None  and geojson != "")

    # load in geojson into json
    geometries  = json.loads(geojson)
    features = geometries['features']

    # make sure that holes are added within the annotation rather than separate features
    # if working, expects 1 feature object containing 1 exterior and 2 interiors (holes)
    assert len(features) == 1
    