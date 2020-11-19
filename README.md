[![MSK-MIND](https://circleci.com/gh/msk-mind/data-processing.svg?style=shield)](https://circleci.com/gh/msk-mind/data-processing) [![codecov](https://codecov.io/gh/msk-mind/data-processing/branch/circleci-project-setup/graph/badge.svg)](https://codecov.io/gh/msk-mind/data-processing)

# Data Processing
Scripts for data processing


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


## API Steps:

Start servers `./start.sh`

Run sequence of API calls:

```

curl --header "Content-Type: application/json" --request POST --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' http://pllimsksparky1:5000/mind/api/v1/transferFiles

curl --header "Content-Type: application/json" --request POST --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' http://pllimsksparky1:5001/mind/api/v1/buildRadiologyProxyTables

curl --header "Content-Type: application/json" --request POST --data '{"TEMPLATE":"/gpfs/mskmindhdp_emc/user-templates/API-TEST-0000_20201120_JnyEQdNNIw.yaml"}' http://pllimsksparky1:5001/mind/api/v1/buildRadiologyGraph

curl --request GET http://pllimsksparky1:5001/mind/api/v1/datasets

curl --request GET http://pllimsksparky1:5001/mind/api/v1/datasets/API-TEST-0000_20201120_JnyEQdNNIw

curl --request GET http://pllimsksparky1:5001/mind/api/v1/datasets/API-TEST-0000_20201120_JnyEQdNNIw/countDicom
```
