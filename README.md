[![MSK-MIND](https://circleci.com/gh/msk-mind/data-processing.svg?style=shield)](https://circleci.com/gh/msk-mind/data-processing) [![codecov](https://codecov.io/gh/msk-mind/data-processing/branch/master/graph/badge.svg)](https://app.codecov.io/gh/msk-mind/data-processing)

# Data Processing
Scripts for data processing

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
