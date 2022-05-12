'''
Created on May 12, 2022

@author: pashaa@mskcc.org
'''


import requests
import json
import argparse
import sys
import logging


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


def create_source(url, token):
  '''
  Create a datalake NAS source.

  :param token: authorization token
  :return:
  '''
  endpoint = url+"/api/v3/source"

  payload = json.dumps({
    "_type": "source",
    "name": "local_lake",
    "description": "tutorial source",
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provision a Dremio instance')
    parser.add_argument(
        '--username', '-u', default=None,  help='admin username')
    parser.add_argument(
        '--password', '-p', default=None, help='admin password')
    parser.add_argument(
        '--port', '-r', default=None, help='dremio api port')
    parser.add_argument(
      '--verbose', '-v', default=0, help='Increase verbosity')

    print('\nProvisioning Dremio...')

    opts = parser.parse_args(args=sys.argv[1:])
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(max(1, logging.WARNING - 10 * opts.verbose))

    url = "http://dremio:"+opts.port

    response = login(url, opts.username, opts.password)

    if response.status_code == 200:
        token = response.json()['token']
        
        response = create_source(url, token)
        
        if response.status_code != 200:
            print('Dremio provision error: {}'.format(response.json()['errorMessage']))
        else:
            print('Successful!\n')
            
    else:
        print('Dremio provision error: unable to log into dremio '
              'using username={} password={}'.format(opts.username, opts.password))



