IMAGE=$(USER)-luna-tutorial:0.0.1
PWD=$(shell pwd)
USER_NAME=$(USER)
#GROUP_NAME=$(shell id -g -n $(USER))
GROUP_NAME=$(USER)
USER_ID=$(shell id -u)
GROUP_ID=$(shell id -g)
CURRENT_UID=$(USER):$(GID)
DOCKER_PATH=$(shell which docker)

export COMPOSE_PROJECT_NAME=$(USER_NAME)

.PHONY: all help build clean test run
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
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


clean:      ## cleanup luna_tutorial images, containers volumes and results files on host
	-docker rmi $$(docker images luna* -q)
	-docker rmi $$(docker container ls -f "name=luna*" -q)
	rm -rf vmount/PRO-12-123
	rm -rf vmount/assetstore
	rm vmount/logs/tutorial.log


clean-image:  ## remove luna tutorial image
	-docker rmi -f $$(docker images luna* -q)


build:        ## build containers
	mkdir -p vmount/logs vmount/db vmount/tmp
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose build --progress plain


run:        ## launch containers
	./apply_ports.sh
	mkdir -p vmount/logs vmount/db vmount/tmp vmount/assetstore
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose up -d
	@sleep 4
	make download_sample_data
	@echo
	@echo Minio is deployed with root credentials admin:password1 at,
	@echo
	@echo "http://$$(hostname):$$(cat .minio_web_port)"
	@echo
	@echo == You may use this interface to access your project files
	@echo
	@echo
	@echo Dremio is deployed at,
	@echo
	@echo "http://$$(hostname):$$(cat .dremio_port)"
	@echo
	@echo == Open Dremio in your browser and create a user with root credentials admin:password1
	@echo == Then provision Dremio by executing \"make provision_dremio\"
	@echo == Refresh you browser and a Data Lake called \"local_lake\" should appear on your Dremio interface
	@echo == You may use this interface to view and query the parquet datasets
	@echo
	@echo
	@echo DSA is deployed and provisioned with root credentials admin:password1 at,
	@echo
	@echo "http://$$(hostname):$$(cat .dsa_port)"
	@echo
	@echo == Execute \"make download_sample_data\" to download sample svs files and import them into DSA. This takes a few minutes. 
	@echo
	@echo
	@echo Finally, Jupyterlab is deployed at,
	@echo
	@echo "http://$$(hostname):$$(cat .jupyter_port)/tree/notebooks"
	@echo
	@echo == Open Jupyterlab in your broswer and copy-paste this token to view and execute the tutorial notebooks
	@echo
	@sleep 4
	@grep "token=" vmount/logs/tutorial.log |  tail -1 | cut -d "=" -f2
	@echo

stop:       ## terminate containers but keep volumes
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose down
	rm -f .*_port

exec:       ## launches terminal prompt inside luna_tutorial container
	docker exec -it $(USER_NAME)_luna_tutorial /bin/bash


provision_dremio:    ## provision dremio
	docker exec $(USER_NAME)_luna_tutorial python vmount/provision_dremio.py -u admin -p password1 -d $$(cat .dremio_port) -m $$(cat .minio_api_port)


download_sample_data:    ## download sample_data
	docker exec $(USER_NAME)_girder python vmount/provision_girder.py --sample-data --user $(USER_NAME)

