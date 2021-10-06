.PHONY: help clean clean-pyc clean-build list test coverage

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "dist - create a distribution"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "grpt - generate radiology proxy table."
	@echo "Usage: make grpt template-file=<data_ingestion_template_file_name> config-file=<config_file> process-string=\"transfer,delta,graph\""

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-test:      ## remove test and coverage artifacts
	rm -fr .pytest_cache
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

dist: clean
	pip install --upgrade pip
	pip install --use-feature=2020-resolver -r requirements_dev.txt
	python setup.py sdist bdist_wheel
	ls -l dist

# https://packaging.python.org/tutorials/packaging-projects/
test-upload:
	python3 -m twine upload --verbose --repository testpypi dist/*.whl

upload:
	python3 -m twine upload dist/*.whl


lint:
	flake8 data-processing test

test: clean-test clean-pyc    ## run tests quickly with the default Python
	pip install --upgrade pip
	pip install --use-feature=2020-resolver -r requirements_dev.txt
	pytest

coverage:
	pytest -s --cov=pyluna-core --cov=pyluna-common --cov=pyluna-radiology --cov=pyluna-pathology .
	coverage report -m
	coverage html
	open htmlcov/index.html

grpt:
	time python3 -m luna.radiology.proxy_table.generate -t $(template-file) -f $(config-file) -p $(process-string)
gppt:
	time python3 -m luna.pathology.proxy_table.generate -d $(template-file) -a $(config-file) -p $(process-string)

