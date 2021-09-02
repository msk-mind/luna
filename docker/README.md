# Luna Docker

## Content

- **Dockerfile**:
Simple image that installs openslide, jre and python dependencies from `requirements_dev.txt`.
This image is built and pushed to dockerhub, and used in circleci tests.
  
- **requirements_dev.txt**:
Python dependencies for luna project
  
## Build and push image to Dockerhub

Building and pushing the docker image to dockerhub can be done via the circleci workflow `run_workflow_docker` (see .circleci/config.yml for more details.)
This workflow is off by default. To trigger this workflow using circleci, first create a [circleci personal token](https://circleci.com/docs/2.0/managing-api-tokens/#creating-a-personal-api-token).
Then send the post request like below, replacing `<your-personal-circleci-api-token>`, `<your-branch>`.

```
curl --request POST \
  --url https://circleci.com/api/v2/project/gh/msk-mind/luna/pipeline \
  --header 'Circle-Token: <your-personal-circleci-api-token>' \
  --header 'content-type: application/json' \
  --data '{"branch": "<your-branch>", "parameters" : {"run_workflow_test": false, "run_workflow_docker":  true }}'
```