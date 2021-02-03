"""
The steps for processing pathology nuclei point annotations includes:
1) (proxy_table) JSON Anotations are first downloaded from slideviewer
2) (refined_table) JSON annotations are then converted into qupath-compatible geojson files
"""

import os, json
import click
from pyspark.sql.window import Window
from pyspark.sql.functions import first, last, col, lit, desc, udf, explode, array, to_json, current_timestamp
from pyspark.sql.types import ArrayType, StringType, MapType, StructType, StructField

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
from data_processing.common.sparksession import SparkConfig
from data_processing.common.utils import generate_uuid_dict
#from data_processing.common.download_annotations 	import download_point_annotation, fetch_slide_tuples
import data_processing.common.constants as const

logger = init_logger()

os.environ['OPENBLAS_NUM_THREADS'] = '1'

def download_sv_point_annotation(url):
    from urllib.request import urlopen
    """
    Return json response from slide viewer call
    :param url: slide viewer api to call
    :return: json response
    """
    print(url)
    response = urlopen(url)
    data = json.load(response)
    print(data)
    print(str(data) != "[]")
    if(str(data) != '[]'):
        return data
    else:
        print(" +- Label annotation file does not exist for slide and user.")
        return None

def download_point_annotation(slideviewer_path, project_id, user):

    print (f" >>>>>>> Processing [{slideviewer_path}] <<<<<<<<")

    url = "https://slides-res.mskcc.org/slides/" + str(user) + "@mskcc.org/projects;" + str(project_id) + ';' + slideviewer_path + "/getSVGLabels/nucleus"

    return download_sv_point_annotation(url)

def fetch_slide_ids(url, project_id, dest_dir, csv_file=None):
    '''
    Fetch the list of slide ids from the slideviewer server for the project with the specified project id.
    Alternately, a slideviewer csv file may be provided to override download from server.
    :param url - slideviewer url. url may be None if csv_file is specified.
    :param csv_file - an optional slideviewer csv file may be provided to override the need to download the file
    :param project_id - slideviewer project id from which to fetch slide ids
    :param dest_dir - directory where csv file should be downloaded
    :return: list of slide ids
    '''

    # run on all slides from specified SLIDEVIEWER_CSV file.
    # if file is not specified, then download file using slideviewer API
    # download entire slide set using project id
    # the file is then written to the landing directory
    import requests
    if csv_file == None or \
            csv_file == '' or not \
            os.path.exists(csv_file):

        csv_file = os.path.join(dest_dir, 'project_' + str(project_id) + '.csv')

        url = url + 'exportProjectCSV?pid={pid}'.format(pid=str(project_id))
        res = requests.get(url)

        with open(csv_file, "wb") as slideoutfile:
            slideoutfile.write(res.content)

    # read slide ids
    slides = []
    with open(csv_file) as slideoutfile:
        # skip first 4 lines
        count = 0
        for line in slideoutfile:
            count += 1
            if count == 4:
                break

        # read whole slide image file names contained in the project in slide viewer
        for line in slideoutfile:
            full_filename = line.strip()
            slidename = full_filename.split(";")[-1].replace(".svs", "")
            slides.append([full_filename, slidename, project_id])

    return slides

@click.command()
@click.option('-d', '--data_config_file', default=None, type=click.Path(exists=True),
              help="path to data yaml file containing information required for pathology point annotation data ingestion. "
                   "See data-config.yaml.template")
@click.option('-a', '--app_config_file', default='config.yaml', type=click.Path(exists=True),
              help="path to config file containing application configuration. See config.yaml.template")
def cli(data_config_file, app_config_file):
    """
    This module generates point_json_raw table for pathology data based on information specified in the template file.

    Example:
        python -m data_processing.pathology.point_annotation.proxy_table.generate \
        --data_config_file {PATH_TO_DATA_CONFIG_FILE} \
        --app_config_file {PATH_TO_APP_CONFIG_FILE}

    """
    with CodeTimer(logger, 'generate POINT_RAW_JSON table'):
        logger.info('data config file: ' + data_config_file)
        logger.info('app config file: ' + app_config_file)

        # load configs
        cfg = ConfigSet(name=const.DATA_CFG, config_file=data_config_file)
        cfg = ConfigSet(name=const.APP_CFG,  config_file=app_config_file)

        create_proxy_table()


def create_proxy_table():

    cfg = ConfigSet()

    spark = SparkConfig().spark_session(config_name=const.APP_CFG, app_name="data_processing.pathology.point_annotation.proxy_table.generate")

    # load paths from configs
    point_table_path = const.TABLE_LOCATION(cfg)

    PROJECT_ID = cfg.get_value(path=const.DATA_CFG+'::PROJECT_ID')

    # Get slide list to use
    slides = fetch_slide_ids(None, PROJECT_ID, '.', cfg.get_value(path=const.DATA_CFG+'::SLIDEVIEWER_CSV_FILE'))
    logger.info(slides)

    schema = StructType([StructField("slideviewer_path", StringType()),StructField("slide_id", StringType()),StructField("sv_project_id", StringType())])
    df = spark.createDataFrame(slides, schema)
    # populate columns
    df = df.withColumn("users", array([lit(user) for user in cfg.get_value(const.DATA_CFG+'::USERS')]))
    df = df.select("slideviewer_path", "slide_id", "sv_project_id", explode("users").alias("user"))
    df.show()
    # download slide point annotation jsons
    # example point json:
    # [{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1440","y":"747","class":"0","classname":"Tissue 1"},{"project_id":"8","image_id":"123.svs","label_type":"nucleus","x":"1424","y":"774","class":"3","classname":"Tissue 4"}]
    point_json_struct = ArrayType(
        MapType(StringType(), StringType())
    )
    download_point_annotation_udf = udf(download_point_annotation,  point_json_struct)
    df = df.withColumn("sv_json", download_point_annotation_udf("slideviewer_path", "sv_project_id", "user")) \
            .dropna(subset=["sv_json"])

    df.show(10, False)

    # populate "date_added", "date_updated","latest", "sv_json_record_uuid"
    sv_json_record_uuid_udf = udf(generate_uuid_dict, StringType())
    spark.sparkContext.addPyFile("./data_processing/common/EnsureByteContext.py")
    df = df.withColumn("sv_json_record_uuid", sv_json_record_uuid_udf(to_json("sv_json"), array(lit("SVPTJSON")))) \
        .withColumn("latest", lit(True)) \
        .withColumn("date_added", current_timestamp()) \
        .withColumn("date_updated", current_timestamp())

    # create proxy sv_point json table
    # update main table if exists, otherwise create main table
    if not os.path.exists(point_table_path):
        df.write.format("delta").save(point_table_path)
    else:
        from delta.tables import DeltaTable
        point_table = DeltaTable.forPath(spark, point_table_path)
        point_table.alias("main_point_table") \
            .merge(df.alias("point_annotation_updates"), "main_point_table.sv_json_record_uuid = point_annotation_updates.sv_json_record_uuid") \
            .whenMatchedUpdate(set = { "date_updated" : "point_annotation_updates.date_updated" } ) \
            .whenNotMatchedInsertAll() \
            .execute()

    # add latest flag
    windowSpec = Window.partitionBy("user", "slide_id").orderBy(desc("date_updated"))
    # Note that last != opposite of first! Have to use desc ordering with first...
    spark.read.format("delta").load(point_table_path) \
        .withColumn("date_latest", first("date_updated").over(windowSpec)) \
        .withColumn("latest", col("date_latest")==col("date_updated")) \
        .drop("date_latest") \
        .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(point_table_path)

if __name__ == "__main__":
    cli()
