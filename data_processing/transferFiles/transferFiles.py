#!/gpfs/mskmindhdp_emc/sw/env/bin/python3

"""
To start a server: ./data_processing_app.py (Recommended on sparky1)
"""

from flask import Flask, request

from common.CodeTimer import CodeTimer
from common.custom_logger import init_logger
from common.sparksession import SparkConfig
from common.Neo4jConnection import Neo4jConnection
import data_processing.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")
# spark = SparkConfig().spark_session(os.environ['SPARK_CONFIG'], "data_processing.mind.api")

"""
Transfer Radiology files
"""

@app.route('//mind/api/v1/transferFiles', methods=['GET', 'POST'])
def transferFiles():
    start_time = time.time()
    setup_environment_from_yaml(os.environ['PATH_TO_TEMPLATE_FILE'])
    transfer_cmd = ["time", "../data_processing/radiology/proxy_table/transfer_files.sh"]

    try:
        exit_code = subprocess.call(transfer_cmd)
        logger.info("--- Finished transferring files in %s seconds ---" % (time.time() - start_time))
    except Exception as err:
        logger.error(("Error Transferring files with rsync" + str(err)))
        return


    if exit_code != 0:
        logger.error(("Error Transfering files - Non-zero exit code: " + str(exit_code)))
        return "Radiology files transfer is unsuccessful"

    return "Radiology files transfer is successful"

def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, 'r') as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var]).strip()



if __name__ == '__main__':
    app.run(host=os.environ['HOSTNAME'], debug=True)
