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

def TABLE_LOCATION(cfg): 
    return "/data/{0}/tables/{1}".format(  cfg.get_value(name=DATA_CFG, jsonpath='PROJECT'), cfg.get_value(name=DATA_CFG, jsonpath='DATASET_NAME' ))
