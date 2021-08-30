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
    os.environ['RAW_DATA_PATH'] = os.getcwd()+\
                                     '/pyluna-radiology/tests/luna/radiology/proxy_table/test_data/destination'


    yield env

    print('------teardown------')
    shutil.rmtree(os.environ.get('RAW_DATA_PATH'))


'''
def test_transfer_files_1(env):
    #
    # Chunk file points to a files and include limits to one file types
    #
    os.environ['CHUNK_FILE'] = 'pyluna-radiology/tests/luna/radiology/proxy_table/test_data/chunk_file1.txt'
    os.environ['SOURCE_PATH'] = os.getcwd() + \
                                '/pyluna-radiology/tests/luna/radiology/proxy_table/test_data/source/'
    os.environ['INCLUDE'] = '--include=*.dcm'
    os.environ['FILE_COUNT'] = '3'
    os.environ['DATA_SIZE'] = '24'

    transfer_cmd = ["time", "./luna/radiology/proxy_table/transfer_files.sh"]

    try:
        exit_code = subprocess.call(transfer_cmd)
    except Exception as err:
        pytest.fail(msg="Error Transfering files with rsync" + str(err))


    assert(exit_code == 0)
    assert os.path.exists(os.getenv('RAW_DATA_PATH'))

    assert os.path.exists(os.getenv('RAW_DATA_PATH')+'/test1.dcm')
    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/test2.dcm')
    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/test3.dcm')

    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/test1.mha')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/test1.mhd')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/test1.raw')

    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/nested_dir')
'''


'''
def test_transfer_files_2(env):
    #
    # Chunk file points to a dir and include limits to two file types
    #
    os.environ['CHUNK_FILE'] = 'pyluna-radiology/tests/luna/radiology/proxy_table/test_data/chunk_file2.txt'
    os.environ['SOURCE_PATH'] = os.getcwd() + \
                                '/pyluna-radiology/tests/luna/radiology/proxy_table/test_data'
    os.environ['INCLUDE'] = '--include=*.dcm --include=*.mha'
    os.environ['FILE_COUNT'] = '6'
    os.environ['DATA_SIZE'] = '48'

    transfer_cmd = ["time", "./luna/radiology/proxy_table/transfer_files.sh"]

    try:
        exit_code = subprocess.call(transfer_cmd)
    except Exception as err:
        pytest.fail(msg="Error Transfering files with rsync" + str(err))


    assert(exit_code == 0)
    assert os.path.exists(os.getenv('RAW_DATA_PATH'))

    assert os.path.exists(os.getenv('RAW_DATA_PATH')+'/source/test1.dcm')
    assert os.path.exists(os.getenv('RAW_DATA_PATH')+'/source/test1.mha')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/source/test1.mhd')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH')+'/source/test1.raw')

    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/test2.dcm')
    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/test2.mha')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/test2.mhd')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/test2.raw')

    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/double_nested_dir/test3.dcm')
    assert os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/double_nested_dir/test3.mha')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/double_nested_dir/test3.mhd')
    assert not os.path.exists(os.getenv('RAW_DATA_PATH') + '/source/nested_dir/double_nested_dir/test3.raw')
'''