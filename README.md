# Data Processing
Scripts for data processing


## Steps to generate radiology proxy table.
1. Make a copy of data_ingestion_template.yaml.template and fill it out. This template contains operational metadata associated with the data and the data transfer process. 

2. Make a copy of the config.yaml.template and fill it out. This config file contains operational metadata associated with the ETL process.

3. Execute grpt command. 

TODO: add instructions for using screen to run the script 

```bash
make grpt template-file=your_data_ingestion_template_file_name 
```   

4. Verify logs.
   
   
TODO: Development steps for radiology proxy table development   

1. arfath - verify data ingestion template against schema (python code, no dependencies)

2. druv - load data_ingestion_template into env_vars (python code)

3. arfath - write transfer bash script with idempotency (depends on 1)

4. doori, andy - write proxy table generation code with idempotency (no dependencies, assumes raw files are local)

5. arfath - write unit tests for data transfer

6. doori, andy - write unit tests for proxy table generation   