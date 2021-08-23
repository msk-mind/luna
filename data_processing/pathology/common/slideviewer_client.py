'''
Created on January 31, 2021

@author: pashaa@mskcc.org

Functions for downloading annotations from SlideViewer
'''
import os, shutil
from typing import Dict
import zipfile
import requests
import logging

logger = logging.getLogger(__name__)

def get_slide_id(full_filename:str) -> str:
    """get slide id
    
    Get slide id from the slideviewer full file name. The full_filename in 
    the slideview csv is of the format: year;HOBS_ID;slide_id.svs
    for example: 2013;HobS13-283072057510;1435197.svs

    Args:
        full_filename (str): full filename of slide 
    
    Returns:    
        str: numeric slide id
    """
    return full_filename.split(";")[-1].replace(".svs", "")


def fetch_slide_ids(url:str, project_id:int, dest_dir:str, csv_file:str=None)->list:
    """get slide ids

    Fetch the list of slide ids from the slideviewer server for the project with the
    specified project id. Alternately, a slideviewer csv file may be provided to 
    override download from server.

    Args:
        url (str or None): slideviewer url. url may be None if csv_file is specified.
        project_id (int): slideviewer project id from which to fetch slide ids
        dest_dir (str): directory where csv file should be downloaded
        csv_file (str): slideviewer csv file may be provided to override the need 
        to download the file
    
    Returns:
        list:  list of (slideviewer_path, slide_id, sv_project_id)
    """

    # run on all slides from specified SLIDEVIEWER_CSV file.
    # if file is not specified, then download file using slideviewer API
    # download entire slide set using project id
    # the file is then written to the dest directory
    new_csv_file = os.path.join(dest_dir, 'project_' + str(project_id) + '.csv')

    if csv_file == None or \
            csv_file == '' or not \
            os.path.exists(csv_file):

        url = url + 'exportProjectCSV?pid={pid}'.format(pid=str(project_id))
        res = requests.get(url)

        with open(new_csv_file, "wb") as slideoutfile:
            slideoutfile.write(res.content)

    else:
        # copy given csv_file to dest directory
        shutil.copy(csv_file, new_csv_file)

    # read slide ids
    slides = []
    with open(new_csv_file) as slideoutfile:
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
            slides.append([full_filename, slidename, project_id])

    return slides


def download_zip(url:str, dest_path:str, chunk_size:int = 128)->bool:
    """Download zip file

    Downloads zip from the specified URL and saves it to the specified file path.
    see https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    
    Args:
        url (str): slideviewer url to download zip from 
        dest_path (str): file path where zipfile should be saved
        chunk_size (int): size in bytes of chunks to batch out during download
    
    Returns:
        bool: True if zipfile downloaded and saved successfully, else false
    """
    
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk == b'Label image not found.':  # message from slideviewer
                return False
            else:
                fd.write(chunk)
        return True

def unzip(zipfile_path:str) -> any:
    """unzip zip file

    Args:
        zipfile_path (str): path of zipfile to unzip
    
    Returns:
        readfile pointer to unzippped file if successfully unzippped, else None
    """
    logger.info("Unzipping " + zipfile_path)
    try:
        return zipfile.ZipFile(zipfile_path)  # returns read file pointer
    except zipfile.BadZipFile as err:
        logger.exception('Dumping invalid Zipfile ' + zipfile_path + ':')
        return None


def download_sv_point_annotation(url:str) -> Dict[str, any]:
    """download slideviwer point annotation

    Calls slideviewer API with the given url

    Args:
        url (str): slide viewer api to call
    
    Returns:
        dict[str, any]: json response
    """
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as err:
        logger.exception("General exception raised while trying " + url)
        return None

    logger.info("Found data = " + str(data))
    if(str(data) != '[]'):
        return data
    else:
        logger.warning("Label annotation file does not exist for slide and user.")
        return None
