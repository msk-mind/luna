IMAGE=$(USER)-luna-tutorial:0.0.1
CONTAINER=$(USER)-luna-tutorial
PWD=$(shell pwd)
USER_NAME=$(USER)
GROUP_NAME=$(shell id -g -n $(USER))
USER_ID=$(shell id -u)
GROUP_ID=$(shell id -g)
CURRENT_UID=$(USER):$(GID)

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


clean:      ## cleanup all images volumes and containers
	docker system prune -a -f
	docker system prune --volumes -f

clean-image:  ## remove luna tutorial image
	docker rmi -f $(IMAGE)

build:      ## build luna tutorial docker image
	DOCKER_BUILDKIT=0 time docker build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) --build-arg USER_NAME=$(USER_NAME) --build-arg GROUP_NAME=$(GROUP_NAME) -t $(IMAGE) .

run:       ## run the luna-tutorial container
	mkdir -p vmount
	docker run -it --name $(CONTAINER) -v $(PWD)/vmount:/home/$(USER_NAME)/vmount:rw -p 127.0.0.1:8888:8888/tcp $(IMAGE)
	docker rm $(CONTAINER)

exec:      ## enter the luna-tutorial container
	docker exec -it $(CONTAINER) /bin/bash

backup:  ## backup notebook changes from container to the host notebooks dir. need to rebuild image to bring in changes
	./backup.sh
