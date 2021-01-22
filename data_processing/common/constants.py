'''
Created on November 16, 2020

@author: rosed2@mskcc.org
'''
import os
# Application Constants

#### Table Names ####
TABLE_DIR			='tables/'
# clinical
DIAGNOSIS_TABLE		=TABLE_DIR + 'diagnosis'
MEDICATION_TABLE	=TABLE_DIR + 'medication'
PATIENT_TABLE		=TABLE_DIR + 'patient'

# radiology
DICOM_TABLE			=TABLE_DIR + 'dicom'
SCAN_TABLE 			=TABLE_DIR + 'scan'
SCAN_ANNOTATION_TABLE		=TABLE_DIR + 'scan_annotation'
FEATURE_TABLE		=TABLE_DIR + 'feature'

#### Raw Data Directories ####
DICOMS				='dicoms'
SCANS				='scans'
SCAN_ANNOTATIONS	='scan_annotations'
FEATURES			='features'

# Configurations
APP_CFG				='APP_CFG'
DATA_CFG			='DATA_CFG'
SCHEMA_FILE			='data_ingestion_template_schema.yml'

PUBLIC_DIR          ='/gpfs/mskmind_ess/mind_public'


def PROJECT_LOCATION(cfg):
    """
    ROOT_PATH is a path to mind data e.g. /gpfs/mind/data or hdfs://server:port/data

    :param cfg:
    :return: ROOT_PATH/PROJECT_NAME
    """
    return os.path.join(cfg.get_value(path=DATA_CFG+'::ROOT_PATH'), cfg.get_value(path=DATA_CFG+'::PROJECT'))


def TABLE_LOCATION(cfg): 
    return "{0}/tables/{1}".format(PROJECT_LOCATION(cfg), TABLE_NAME(cfg))

def TABLE_NAME(cfg):
    return "{0}_{1}".format(cfg.get_value(path=DATA_CFG+'::DATA_TYPE').upper(), cfg.get_value(path=DATA_CFG+'::DATASET_NAME'))
