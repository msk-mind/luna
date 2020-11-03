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
   