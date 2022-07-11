'''
Created on May 12, 2022

@author: pashaa@mskcc.org
'''


import requests
import json
import argparse
import sys
import logging
import socket

logger = logging.getLogger(__name__)
# See http://docs.python.org/3.3/howto/logging.html#configuring-logging-for-a-library
logging.getLogger(__name__).addHandler(logging.StreamHandler)


def login(url, username, password):
  ''':param username
     :param password
  '''
  endpoint = url+"/apiv2/login"

  payload = json.dumps({
    "userName": username,
    "password": password,
  })
  headers = {
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", endpoint, headers=headers, data=payload)

  return response


def create_nas_source(url, token):
  '''
  Create a datalake NAS source.

  :param token: authorization token
  :return:
  '''
  endpoint = url+"/api/v3/source"

  payload = json.dumps({
    "_type": "source",
    "name": "local_lake",
    "description": "tutorial data lake",
    "type": "NAS",
    "config": {
      "path": "/data"
    },
    "metadataPolicy": {
      "authTTLMs": 60000,
      "datasetRefreshAfterMs": 60000,
      "datasetExpireAfterMs": 60000,
      "namesRefreshMs": 60000,
      "datasetUpdateMode": "PREFETCH"
    },
    "accelerationRefreshPeriodMs": 60000,
    "accelerationGracePeriodMs": 60000
  })
  headers = {
    'Authorization': 'Bearer '+token,
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", endpoint, headers=headers, data=payload)

  return response


def create_s3_source(dremio_url, minio_url, token):
  '''
  Create a datalake S3 source.

  :param url: URL of S3 source
  :param token: authorization token
  :return:
  '''
  print('Creating S3 source to {}...'.format(minio_url))

  endpoint = dremio_url+"/api/v3/source"

  payload = json.dumps({
    "_type": "source",
    "name": "local_s3_lake",
    "description": "tutorial data lake",
    "type": "S3",
    "config": {
      "accessKey": "admin",
      "accessSecret": "password1",
      "secure": "false",
      "externalBucketList": [],
      "rootPath": "/",
      "compatibilityMode": "true",
      "propertyList": [
         {"name": "fs.s3a.path.style.access", "value": "true"},
         {"name": "fs.s3a.endpoint",          "value": minio_url}
      ]
    },
    "metadataPolicy": {
      "authTTLMs": 60000,
      "datasetRefreshAfterMs": 60000,
      "datasetExpireAfterMs": 60000,
      "namesRefreshMs": 60000,
      "datasetUpdateMode": "PREFETCH"
    },
    "accelerationRefreshPeriodMs": 60000,
    "accelerationGracePeriodMs": 60000
  })
  headers = {
    'Authorization': 'Bearer '+token,
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", endpoint, headers=headers, data=payload)

  return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provision a Dremio instance')
    parser.add_argument(
        '--username', '-u', default=None,  help='admin username')
    parser.add_argument(
        '--password', '-p', default=None, help='admin password')
    parser.add_argument(
        '--dremio_port', '-d', default=None, help='dremio api port')
    parser.add_argument(
        '--minio_port', '-m', default=None, help='minio api port')
    parser.add_argument(
      '--verbose', '-v', default=0, help='Increase verbosity')

    print('\nProvisioning Dremio...')

    opts = parser.parse_args(args=sys.argv[1:])
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(max(1, logging.WARNING - 10 * opts.verbose))

    dremio_url = "http://dremio:"+opts.dremio_port
    minio_url = socket.gethostbyname('minio')+':'+opts.minio_port  # only works with ip address


    response = login(dremio_url, opts.username, opts.password)

    print('Logging in to Dremio...')

    if response.status_code == 200:
        token = response.json()['token']
        
        response = create_s3_source(dremio_url, minio_url, token)
        
        if response.status_code != 200:
            print('Dremio provision error: {}'.format(response.json()['errorMessage']))
        else:
            print('Successful!\n')
            
    else:
        print('Dremio provision error: unable to log into dremio '
              'using username={} password={}'.format(opts.username, opts.password))



