'''
Created on January 31, 2021

@author: pashaa@mskcc.org

Functions for downloading regional annotation bmps from SlideViewer
'''
import os
import requests
from data_processing.common.config import ConfigSet
from data_processing.common.constants import DATA_CFG


def get_slide_id(full_filename):
    '''
    Get slide id from the slideviewer full file name. The full_filename in the slideview csv is of the format:

    year;HOBS_ID;slide_id.svs

    for example: 2013;HobS13-283072057510;1435197.svs

    :param full_filename: full filename
    :return: slide id
    '''
    return full_filename.split(";")[-1].replace(".svs", "")


def fetch_slide_ids():
    '''
    Fetch the list of slide ids for the project from SlideViewer.

    :return: list of slide ids
    '''
    cfg = ConfigSet()
    SLIDEVIEWER_API_URL = cfg.get_value(path=DATA_CFG + '::SLIDEVIEWER_API_URL')
    SLIDEVIEWER_CSV_FILE = cfg.get_value(path=DATA_CFG + '::SLIDEVIEWER_CSV_FILE')
    PROJECT_ID = cfg.get_value(path=DATA_CFG + '::PROJECT_ID')
    LANDING_PATH = cfg.get_value(path=DATA_CFG + '::LANDING_PATH')

    # run on all slides from specified SLIDEVIEWER_CSV file.
    # if file is not specified, then download file using slideviewer API
    # download entire slide set using project id
    # the file is then written to the landing directory
    if SLIDEVIEWER_CSV_FILE == None or \
            SLIDEVIEWER_CSV_FILE == '' or not \
            os.path.exists(SLIDEVIEWER_CSV_FILE):

        SLIDEVIEWER_CSV_FILE = os.path.join(LANDING_PATH,'project_'+str(PROJECT_ID)+'.csv')

        url = SLIDEVIEWER_API_URL+'exportProjectCSV?pid={pid}'.format(pid=str(PROJECT_ID))
        res = requests.get(url)

        with open(SLIDEVIEWER_CSV_FILE, "wb") as slideoutfile:
            slideoutfile.write(res.content)

    # read slide ids
    slides = []
    with open(SLIDEVIEWER_CSV_FILE) as slideoutfile:
        # skip first 4 lines
        count = 0
        for line in slideoutfile:
            count += 1
            if count == 4:
                break

        # read whole slide image file names contained in the project in slide viewer
        for line in slideoutfile:
            full_filename = line.strip()
            slidename = get_slide_id(full_filename)
            slides.append([full_filename, slidename])

    return slides