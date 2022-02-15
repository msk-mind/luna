===============
Developer Guide
===============

A comprehensive guide for those interested in making contributions or
supporting the development of Luna. 

.. warning::
    This documentation is still a work in progress


Development Setup Instructions
==============================
Follow the instructions below to install the latest version of Luna and 
setup a development environment. 

Development Installation
------------------------

Developers interested in using the newest features of Luna, or are
actively contributing to Luna should install the latest version available via::

    git clone git@github.com:msk-mind/luna.git


Download the dependencies with::

    pip install --upgrade pip
    pip install numpy
    cd luna
    pip install -r requirements_dev.txt


For development and testing purposes, add the subpackages to your `PYTHONPATH`::

    export PYTHONPATH=.:pyluna-core:pyluna-common:pyluna-radiology:pyluna-pathology

OR use``setup_local.sh`` to setup your python paths and ``LUNA_HOME`` config::

    source setup_local.sh


Development with Docker
-----------------------

`luna-dev <https://hub.docker.com/r/mskmind/luna-dev>`_ docker image is available on DockerHub.

This docker image includes the pre-requisites and python dependencies.
This is primarily used for circleci testing at the moment, but can be extended based on your development needs.

Once we have a stable release, docker images that includes pyluna packages can be made available.


Documentation Generation
------------------------

API documentation in Luna is generated via ``sphinx-autodoc``. Docstrings are 
written according to the `Google Python Style Guide <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_. 

In order to generate the documentation::
    
    cd docs
    # generate module docs
    sphinx-apidoc --implicit-namespaces -o ./common ../pyluna-common ../pyluna-common/tests ../pyluna-common/setup*
    sphinx-apidoc --implicit-namespaces -o ./pathology ../pyluna-pathology ../pyluna-pathology/tests ../pyluna-pathology/setup*

    # clean up duplicate modules
    # custom common/common-modules.rst pathology/pathology-modules.rst are used.
    rm common/modules.rst pathology/modules.rst
    rm common/pyluna-common.rst pathology/pyluna-pathology.rst

    # generate html
    make html

    # clean _build directory 
    make clean

.. mdinclude:: ../CONTRIBUTING.md

Technical Architecture
======================

Here we provide an overview of the Luna technical architecture and design methodologies. 

Resources
=========

Here are some links to useful developer resources:

- reStructuredText_ for Sphinx_
- pytest_ 

.. _Sphinx: http://sphinx.pocoo.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _pytest: http://docs.pytest.org/en/latest/

