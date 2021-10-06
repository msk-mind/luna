import pytest
import os, shutil
from pyspark import SQLContext
from click.testing import CliRunner
import os
from PIL import Image

from luna.common.config import ConfigSet
from luna.common.sparksession import SparkConfig
from luna.radiology.refined_table.annotation.generate import cli
import luna.common.constants as const

project_path = "pyluna-radiology/tests/luna/radiology/testdata/test-project"
png_table_path = project_path + "/tables/PNG_dsn"
png_config_path = project_path + "/configs/PNG_dsn"
app_config_path = png_config_path + "/app_config.yaml"
data_config_path = png_config_path + "/data_config.yaml"

@pytest.fixture(autouse=True)
def spark():
    print('------setup------')
    ConfigSet(name=const.APP_CFG, config_file='pyluna-radiology/tests/test_config.yml')
    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name='test-radiology-refined-annotation')

    yield spark

    print('------teardown------')
    if os.path.exists(png_table_path):
        shutil.rmtree(png_table_path)
    if os.path.exists(project_path+"/configs"):
        shutil.rmtree(project_path+"/configs")


def test_cli_crop(spark):

    runner = CliRunner()
    result = runner.invoke(cli,
                           ['-d', 'pyluna-radiology/tests/luna/radiology/refined_table/annotation/data_crop.yaml',
                            '-a', 'pyluna-radiology/tests/test_config.yml'])

    assert result.exit_code == 0

    assert os.path.exists(app_config_path)
    assert os.path.exists(data_config_path)

    df = spark.read.format("delta").load(png_table_path)
    dicom_binary = df.select("dicom").head()["dicom"]

    # check that image binary can be loaded with expected width/height
    Image.frombytes("L", (256, 256), bytes(dicom_binary))

    df.unpersist()
