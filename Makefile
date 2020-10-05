.PHONY: help clean clean-pyc clean-build list test coverage

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "dist - create a distribution"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"

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
	pip install -r requirements.txt
	pyinstaller -F --clean --hidden-import py4j.java_collections --exclude-module tkinter data_processing/preprocess_feature.py
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist


lint:
	flake8 data-processing test

test: clean-test clean-pyc    ## run tests quickly with the default Python
	pip install -r requirements.txt
	py.test

coverage:
	pytest -s --cov=src test
	coverage report -m
	coverage html
	open htmlcov/index.html

