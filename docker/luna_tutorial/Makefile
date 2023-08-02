IMAGE=$(USER)-luna-tutorial:0.0.1
PWD=$(shell pwd)
USER_NAME=$(USER)
#GROUP_NAME=$(shell id -g -n $(USER))
GROUP_NAME=$(USER)
USER_ID=$(shell id -u)
GROUP_ID=$(shell id -g)
CURRENT_UID=$(USER):$(GID)
DOCKER_PATH=$(shell which docker)

DREMIO_PORT=9047
MINIO_API_PORT=9000

export COMPOSE_PROJECT_NAME=$(USER_NAME)

.PHONY: all help build clean clean-services clean-images test run summary \
        provision_dremio download_sample_data
.DEFAULT_GOAL := help

# help menu
define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s ... %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


clean: clean-services clean-images    ## delete all tutorial artifacts

clean-services:    ## delete docker containers, sample data and other tutorial artifacts
	rm -rf vmount/PRO-12-123
	rm -rf vmount/assetstore
	rm -fr vmount/logs
	rm -fr vmount/tmp
	rm -fr vmount/db


clean-images:    ## delete the docker images
	-docker image rm -f $$(docker image ls --filter="reference=$(USER_NAME)_luna*" -q)


build:    ## build the docker images
	mkdir -p vmount/logs vmount/db vmount/tmp
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose build 


run:    ## launch the docker containers (minio, dremio, dsa, etc.)
	./apply_ports.sh
	mkdir -p vmount/logs vmount/db vmount/tmp vmount/assetstore
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose up -d
	@sleep 5
	make download_sample_data
	@sleep 2
	make summary


run-tutorial:    ## launch just the main jupyter container
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose up -d luna_tutorial

stop-tutorial:    ## launch just the main jupyter container
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose down luna_tutorial

summary:    ## describe how to connect to the running services
	@echo
	@echo Minio is deployed with credentials \"admin\":\"password1\" at
	@echo
	@echo "    http://$$(hostname):$$(cat .minio_web_port)"
	@echo
	@echo == You can use this interface to access your project files
	@echo
	@echo
	@echo Dremio is deployed at
	@echo
	@echo "    http://$$(hostname):$$(cat .dremio_port)"
	@echo
	@echo == Open Dremio in your browser and register a new user with credentials \"admin\":\"password1\"
	@echo == Once the admin user is created, provision Dremio by executing \"make provision_dremio\"
	@echo
	@echo == Refresh you browser and a Data Lake called \"local_lake\" should appear on your Dremio interface
	@echo == You can use this interface to view and query the parquet datasets
	@echo
	@echo
	@echo DSA is deployed with credentials \"admin\":\"password1\" at
	@echo
	@echo "    http://$$(hostname):$$(cat .dsa_port)"
	@echo
	@echo == Sample svs files should be automatically downloaded and imported into DSA. This takes a few minutes. 
	@echo
	@echo
	@echo Finally, Jupyter is deployed at
	@echo
	@echo "    http://$$(hostname):$$(cat .jupyter_port)/tree/notebooks"
	@echo
	@echo == Open Jupyter in your browser using the link above and the token shown below. to execute the tutorial notebooks.
	@echo
	@echo "   " $$(grep "token=" vmount/logs/tutorial.log | tail -1 | cut -d "=" -f2)
	@echo

stop:    ## terminate containers but keep volumes
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose down
	rm -f .*_port

terminal:    ## open a terminal prompt inside the luna_tutorial container
	docker exec -it $(USER_NAME)_luna_tutorial /bin/bash

exec: terminal

provision_dremio:    ## create a local data lake in dremio for the slde data and annotations
	docker exec $(USER_NAME)_luna_tutorial vmount/conda_python vmount/provision_dremio.py -u admin -p password1 -d $(DREMIO_PORT) -m $(MINIO_API_PORT)

download_sample_data:    ## download sample slides for processing
	@mkdir -p vmount/logs
	@echo "Downloading sample data..."
	@docker exec $(USER_NAME)_girder python3 vmount/provision_girder.py --sample-data --user $(USER_NAME) 2>&1 > vmount/logs/provision_girder.log
	@echo "[download logs are in vmount/logs/provision_girder.log]"

