IMAGE=$(USER)-luna-tutorial:0.0.1
CONTAINER=$(USER)-luna-tutorial
PWD=$(shell pwd)
USER_NAME=$(USER)
#GROUP_NAME=$(shell id -g -n $(USER))
GROUP_NAME=$(USER)
USER_ID=$(shell id -u)
GROUP_ID=$(shell id -g)
CURRENT_UID=$(USER):$(GID)
HOST_IP=127.0.0.1
HOST_PORT=8888
DOCKER_PATH=$(shell which docker)

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
	docker rmi $$(docker images luna* -q)
	docker rmi $$(docker container ls -f "name=luna*" -q)


clean-image:  ## remove luna tutorial image
	docker rmi -f $$(docker images luna* -q)


build:        ## build containers
	apply_ports.sh
	mkdir -p vmount/logs vmount/db vmount/tmp
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) HOST_IP=$(HOST_IP) HOST_PORT=$(HOST_PORT) docker-compose --verbose build


run:        ## launch containers. Default IP:PORT=127.0.0.1:8888. Optionally specify args HOST_IP=xxx.xxx.xxx.xxx HOST_PORT=xxxx.
	mkdir -p vmount/logs vmount/db vmount/tmp
	USER_ID=$(USER_ID) USER_NAME=$(USER_NAME) GROUP_NAME=$(GROUP_NAME) GROUP_ID=$(GROUP_ID) PWD=$(PWD) DOCKER_PATH=$(DOCKER_PATH) HOST_IP=$(HOST_IP) HOST_PORT=$(HOST_PORT) docker-compose --log-level INFO --verbose up -d


stop:       ## terminate containers but keep volumes
	docker-compose down


exec:       ## launches terminal prompt inside luna_tutorial container
	docker exec -it luna_tutorial_luna_tutorial_1 /bin/bash

