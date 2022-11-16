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

The `Makefile` has a number of relevant targets. (Note: GNU Make 3.82 or higher is required.)

See all targets::

    make help

Install virtual environment::

    make venv

OR use``setup_local.sh`` to setup your python paths and ``LUNA_HOME`` config::

    source ./setup_local.sh


Development with Docker
-----------------------

Build docker image::

    make build-docker

The docker image is also available on DockerHub: `luna-dev <https://hub.docker.com/r/mskmind/luna-dev>`_.

This docker image includes the pre-requisites and python dependencies.
This is primarily used for circleci testing at the moment, but can be extended based on your development needs.

Once we have a stable release, docker images that includes pyluna packages can be made available.


Documentation Generation
------------------------

API documentation in Luna is generated via ``sphinx-autodoc``. Docstrings are
written according to the `Google Python Style Guide <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

In order to generate the documentation::

    # generate module docs
    make api-docs

    # activate the virtual env
    conda activate .venv/luna

    # install pandoc if necessary
    mamba install pandoc

    cd docs

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
- docker_

.. _Sphinx: http://sphinx.pocoo.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _docker: https://www.docker.com/

