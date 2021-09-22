===============
Developer Guide
===============

Development Setup Instructions
==============================


Installation
------------

Developers interested in using the newest features of Luna, or are
actively contributing to Luna should install the latest version available via::

    git clone git@github.com:msk-mind/luna.git

For development and testing purposes, add the subpackages to your `PYTHONPATH`::

    export PYTHONPATH=.:src:pyluna-common:pyluna-radiology:pyluna-pathology

OR use ``set_local.sh`` to setup your python paths and ``LUNA_HOME`` config. 

To run tests, specify the subpackage you want to test. For example, this command
will run all tests under ``pyluna-common``::
    
    pytest pyluna-common

Documentation Generation
------------------------

API documentation in Luna is generated via ``sphinx-autodoc``. Docstrings are 
written according to the `Google Python Style Guide <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_. 

In order to generate the documentation::
    
    cd docs
    # generate module docs
    sphinx-apidoc --implicit-namespaces -o ./common ../pyluna-common ../pyluna-common/tests ../pyluna-common/setup*
    sphinx-apidoc --implicit-namespaces -o ./pathology ../pyluna-pathology ../pyluna-pathology  /tests ../pyluna-pathology/setup*

    # generate html
    make html

    # clean _build directory 
    make clean


Contributing
============

If you're interested in contributing to Luna 

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

