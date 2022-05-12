IMAGE=$(USER)-luna-tutorial:0.0.1
CONTAINER=$(USER)-luna-tutorial
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


clean:      ## cleanup luna_tutorial images and containers
	-docker rmi $$(docker images luna* -q)
	-docker rmi $$(docker container ls -f "name=luna*" -q)


clean-image:  ## remove luna tutorial image
	-docker rmi -f $$(docker images luna* -q)


build:        ## build containers
	mkdir -p vmount/logs vmount/db vmount/tmp
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose build --parallel


run:        ## launch containers
	./apply_ports.sh
	mkdir -p vmount/logs vmount/db vmount/tmp vmount/assetstore
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose up -d
	@echo
	@echo Minio is deployed with root credentials admin:password at,
	@echo
	@echo "$$(hostname):$$(cat .minio_port)"
	@echo
	@echo
	@echo Dremio is deployed at,
	@echo
	@echo @echo "$$(hostname):$$(cat .minio_port)"
	@echo
	@echo -> Open Dremio in your browser and create user with root credentials admin:password1
	@echo -> Then provision Dremio by running 'make provision_dremio'
	@echo
	@echo
	@echo DSA is being deployed and provisioned with root credentials admin:password
	@echo -> DSA is being provisioned with sample images and annotations and this will take a few minutes...
	@echo
	@echo "$$(hostname):$$(cat .dsa_port)"
	@echo
	@echo
	@echo Finally, Jupyterlab is deployed at,
	@echo
	@echo "$$(hostname):$$(cat .jupyter_port)"
	@echo
	@echo -> Open Jupyterlab in your broswer and copy-paste this token to view the tutorial notebooks
	@echo
	@sleep 4
	@grep "token=" vmount/logs/tutorial.log |  tail -1 | cut -d "=" -f2
	@echo

stop:       ## terminate containers but keep volumes
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose down
	rm -f .dsa_port
	rm -f .jupyter_port

exec:       ## launches terminal prompt inside luna_tutorial container
	docker exec -it $(USER_NAME)_luna_tutorial_1 /bin/bash


provision_dremio:    ## provision dremio
	docker exec $(USER_NAME)_luna_tutorial_1 python vmount/provision_dremio.py -u admin -p password1 -r $$(cat .dremio_port)

