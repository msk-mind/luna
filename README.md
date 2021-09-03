# Luna
[![MSK-MIND](https://circleci.com/gh/msk-mind/luna.svg?style=shield)](https://circleci.com/gh/msk-mind/luna) [![codecov](https://codecov.io/gh/msk-mind/luna/branch/master/graph/badge.svg)](https://app.codecov.io/gh/msk-mind/luna)
Transformation functions and services for multi-modal oncology data

## Installation
`pip install --upgrade pip`

`pip install numpy`

pyluna can be installed with [all], [radiology], [pathology], and [dev] extra requirements.
> **Note**: for pyluna-* packages that are not on pypi, add your local path for installation to work correctly. Refer to the setup.cfg in the pyluna-* packages for more details.

`pip install .[pathology]` # for pathology dependencies.

Check installation, by importing luna package and printing the version.

`python -c 'import luna; print(luna.__version__)'`

## Development Installation

Install all python dependencies with:

`pip install -r requirements_dev.txt`

For development and testing purposes, add the subpackages to your `PYTHONPATH`:

`export PYTHONPATH=.:src:pyluna-common:pyluna-radiology:pyluna-pathology`

OR use `setup_local.sh` to setup your python paths and LUNA_HOME config:

To run tests, specify the subpackage you want to test. For example, this command will run all tests under pyluna-common package.

`pytest pyluna-common`

## Config

### Global configuration for datastore
The Datastore class interfaces with a variety of backends, currently a NFS/POSIX filesystem, Neo4j and Minio. The class will preferentially pull configuration from `conf/datastore.cfg` first and then fallback to `confg/datastore.default.yml`, which only enables file-based storage.
In this file, you can configure the class to write data/metadata to your common file backend, a neo4j backend, and/or a minio backend, if available and desired. 

### Logging configuration
There is also configuration for centralized logging to MongoDB. Similarily, the init_logger function will preferentially pull configuration from `conf/logging.cfg` first and then fallback to `confg/logging.default.yml`, where by default central logging is turned off

### Setup $LUNA_HOME with conf folder

1. Setup $LUNA_HOME environment variable to point to a location where luna configs can be stored.

``export LUNA_HOME=~/.luna_home``

2. Copy `conf/` folder to $LUNA_HOME/conf

``cp -r conf/ $LUNA_HOME/conf``

3. In the `conf/` folder, copy `logging.default.yml` to `logging.cfg` and `datastore.default.yml` to `datastore.cfg` and modify the `.cfg` files.

``cd $LUNA_HOME/conf``

``cp logging.default.yml logging.cfg``

``cp datastore.default.yml datastore.cfg``

## Steps to generate Sphinx doc pages

```
cd docs

# generate module docs
sphinx-apidoc --implicit-namespaces -o ./common ../pyluna-common ../pyluna-common/tests ../pyluna-common/setup*
sphinx-apidoc --implicit-namespaces -o ./pathology ../pyluna-pathology ../pyluna-pathology/tests ../pyluna-pathology/setup*

# generate html
make html
```

## Steps to generate radiology proxy table.
1. Make a copy of data_ingestion_template.yaml.template and fill it out. This template contains operational metadata associated with the data and the data transfer process. 

2. Make a copy of the config.yaml.template and fill it out. This config file contains operational metadata associated with the ETL process.

3. Execute grpt command. 

NOTE: To kill the job, kill process group, not pid to kill the process & subprocesses

```bash
make grpt template-file=<data_ingestion_template_file_name> \
config-file=<config_file> \
process-string=<any combination of 'transfer,delta,graph'>
```   

4. Verify logs.


## get-path API
```
docker build . -t getpath
docker run -it -p 5002:5002 -e GRAPH_URI=neo4j://localhost:7687 getpath:latest
```
Test with `http://localhost:5002/mind/api/v1/getSlideIDs/case/<accession number>`


## API Steps:

Start servers `./start.sh`

Run sequence of API calls:

```

curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' 
  http://<server>:5000/mind/api/v1/transferFiles

curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' 
  http://<server>:5001/mind/api/v1/buildRadiologyProxyTables

curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' 
  http://<server>:5001/mind/api/v1/buildRadiologyGraph

curl --request GET http://<server>:5001/mind/api/v1/datasets

curl --request GET http://<server>:5001/mind/api/v1/datasets/API-TEST-0000_20201120_JnyEQdNNIw

curl --request GET http://<server>:5001/mind/api/v1/datasets/API-TEST-0000_20201120_JnyEQdNNIw/countDicom
```
