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

export COMPOSE_PROJECT_NAME=$(USER_NAME)-luna-tutorial-

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
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose build


run:        ## launch containers
	./apply_ports.sh
	mkdir -p vmount/logs vmount/db vmount/tmp vmount/assetstore
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose up -d
	@echo
	@echo "DSA is being set up with sample images, annotations and"
	@echo "admin user (admin/password). It will take a couple minutes to become available at,"
	@echo
	@echo "$$(hostname):$$(cat .dsa_port)"
	@echo
	@echo "To access the tutorial, copy-paste this URL below in your browser."
	@echo
	@echo "$$(hostname):$$(cat .jupyter_port)"
	@echo
	@echo "Then copy-paste the token below into your browser page."
	@echo
	@sleep 1
	@cat vmount/logs/tutorial.log | grep "token=" | tail -1

stop:       ## terminate containers but keep volumes
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) docker-compose down
	rm -f .dsa_port
	rm -f .jupyter_port

exec:       ## launches terminal prompt inside luna_tutorial container
	docker exec -it luna_tutorial_luna_tutorial_1 /bin/bash

