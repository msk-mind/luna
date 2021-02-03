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


def fetch_slide_ids(url, project_id, dest_dir, csv_file=None):
    '''
    Fetch the list of slide ids from the slideviewer server for the project with the specified project id.

    Alternately, a slideviewer csv file may be provided to override download from server.

    :param url - slideviewer url. url may be None if csv_file is specified.
    :param csv_file - an optional slideviewer csv file may be provided to override the need to download the file
    :param project_id - slideviewer project id from which to fetch slide ids
    :param dest_dir - directory where csv file should be downloaded

    :return: list of slide ids
    '''

    # run on all slides from specified SLIDEVIEWER_CSV file.
    # if file is not specified, then download file using slideviewer API
    # download entire slide set using project id
    # the file is then written to the landing directory
    if csv_file == None or \
            csv_file == '' or not \
            os.path.exists(csv_file):

        csv_file = os.path.join(dest_dir, 'project_' + str(project_id) + '.csv')

        url = url + 'exportProjectCSV?pid={pid}'.format(pid=str(project_id))
        res = requests.get(url)

        with open(csv_file, "wb") as slideoutfile:
            slideoutfile.write(res.content)

    # read slide ids
    slides = []
    with open(csv_file) as slideoutfile:
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