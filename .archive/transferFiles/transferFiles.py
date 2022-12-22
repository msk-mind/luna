from flask import Flask, request

from luna.common.CodeTimer import CodeTimer
from luna.common.custom_logger import init_logger
from luna.common.sparksession import SparkConfig
from luna.common.Neo4jConnection import Neo4jConnection
import luna.common.constants as const

from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType

import pydicom
import os, shutil, sys, importlib, json, yaml, subprocess, time
from io import BytesIO
from filehash import FileHash
from distutils.util import strtobool
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
logger = init_logger("flask-mind-server.log")

### swagger specific ###
SWAGGER_URL = "/mind/api/v1/docs"
API_URL = "/static/swagger.json"
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "buildRadiologyProxyTables"}
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###

# ==================================================================================================
# Service functions (should be factored out!!!!!!!)
# ==================================================================================================
def setup_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, "r") as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info("Setting up environment:")
    logger.info(template_dict)

    # add all fields from template as env variables
    for var in template_dict:
        os.environ[var] = str(template_dict[var]).strip()


def teardown_environment_from_yaml(template_file):
    # read template_file yaml and set environmental variables for subprocesses
    with open(template_file, "r") as template_file_stream:
        template_dict = yaml.safe_load(template_file_stream)

    logger.info("Tearing down enviornment")

    # delete all fields from template as env variables
    for var in template_dict:
        del os.environ[var]


# ==================================================================================================
# Routes
# ==================================================================================================
"""
Example request:
curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"TEMPLATE":"path/to/template.yaml"}' \
  http://<server>:5000/mind/api/v1/transferFiles
"""


@app.route("//mind/api/v1/transferFiles", methods=["POST"])
def transferFiles():
    data = request.json
    if not "TEMPLATE" in data.keys():
        return "You must supply a template file."
    transfer_cmd = ["time", "./luna/radiology/proxy_table/transfer_files.sh"]

    setup_environment_from_yaml(data["TEMPLATE"])
    with CodeTimer(logger, "setup proxy table"):
        try:
            exit_code = subprocess.call(transfer_cmd)
        except Exception as err:
            logger.error(("Error Transferring files with rsync" + str(err)))
            return "Script failed"
    teardown_environment_from_yaml(data["TEMPLATE"])

    if exit_code != 0:
        logger.error(
            (
                "Some errors occured with transfering files, non-zero exit code: "
                + str(exit_code)
            )
        )
        return "Some errors occured with transfering files, non-zero exit code: " + str(
            exit_code
        )

    return "Radiology files transfer is successful"


if __name__ == "__main__":
    app.run(host=os.environ["HOSTNAME"], port=5000, debug=True)
