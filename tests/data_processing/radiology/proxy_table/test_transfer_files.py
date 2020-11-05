'''
Created on November 04, 2020

@author: pashaa@mskcc.org
'''
import os
import shutil
import subprocess

import pytest

'''
NOTE: These unit tests have been deliberately commented out.
 
To execute these unit tests, follow these steps to allow localhost rsync and then uncomment tests to run them
 
1. Copy your public key (typically at ~/.ssh/is_rsa.pub) to your authorized_keys file (~/.ssh/authorized_keys)

2. If your private key has a passphrase associated with it, override it using the ssh-agent, but first check if the  agent if running on your machine  with

$ eval `ssh-agent`

If it is running, then add the ssh key to the agent with ssh-add and enter your passphrase when prompted for it.
$ ssh-add

Now the terminal will not require a passhprase for any subsequent executions

3. If the ssh-agent is not running, then start the ssh-agent with the command below

$ ssh-agent bash
'''



@pytest.fixture(autouse=True)
def env():
    print('------setup------')
    os.environ['BWLIMIT'] = '1G'
    os.environ['HOST'] = '127.0.0.1'
    os.environ['DESTINATION_PATH'] = os.getcwd()+\
                                     '/tests/data_processing/radiology/proxy_table/test_data/destination'


    yield env

    print('------teardown------')
    #shutil.rmtree(os.environ.get('DESTINATION_PATH'))


'''
def test_transfer_files(env):
    os.environ['CHUNK_FILE'] = 'tests/data_processing/radiology/proxy_table/test_data/chunk_file1.txt'
    os.environ['SOURCE_PATH'] = os.getcwd() + \
                                '/tests/data_processing/radiology/proxy_table/test_data/source'
    os.environ['FILE_COUNT'] = '4'
    os.environ['DATA_SIZE'] = '32'

    transfer_cmd = ["time", "./data_processing/radiology/proxy_table/transfer_files.sh"]

    try:
        exit_code = subprocess.call(transfer_cmd)
    except Exception as err:
        pytest.fail(msg="Error Transfering files with rsync" + str(err))


    assert(exit_code == 0)
    assert os.path.exists(os.getenv('DESTINATION_PATH'))
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/test1.dcm')
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/test1.mha')
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/test1.mhd')
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/test1.raw')
'''


def test_transfer_files(env):
    os.environ['CHUNK_FILE'] = 'tests/data_processing/radiology/proxy_table/test_data/chunk_file2.txt'
    os.environ['SOURCE_PATH'] = os.getcwd() + \
                                '/tests/data_processing/radiology/proxy_table/test_data'
    os.environ['EXCLUDES'] = 'raw,mhd'
    os.environ['FILE_COUNT'] = '2'
    os.environ['DATA_SIZE'] = '16'

    transfer_cmd = ["time", "./data_processing/radiology/proxy_table/transfer_files.sh"]

    try:
        exit_code = subprocess.call(transfer_cmd)
    except Exception as err:
        pytest.fail(msg="Error Transfering files with rsync" + str(err))


    assert(exit_code == 0)
    assert os.path.exists(os.getenv('DESTINATION_PATH'))
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/source/test1.dcm')
    assert os.path.exists(os.getenv('DESTINATION_PATH')+'/source/test1.mha')
    assert not os.path.exists(os.getenv('DESTINATION_PATH')+'/source/test1.mhd')
    assert not os.path.exists(os.getenv('DESTINATION_PATH')+'/source/test1.raw')