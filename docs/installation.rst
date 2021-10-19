.. highlight:: shell

============
Installation
============

See below for installation instructions

1. Pre-requisites
-----------------

- `Java <https://adoptopenjdk.net/installation.html>`_
- `Python3 <https://www.python.org/downloads>`_
- `Openslide <https://openslide.org/download>`_

On Ubuntu, you could install the pre-reqs with::

    apt-get install -y openslide-tools python-openslide default-jre


2. Install with Pip
-------------------

First, update your pip and install numpy (pyradiomics installation fails if numpy is not installed)::

    pip install --upgrade pip
    pip install numpy

To install luna and its basic functionality, run this command in your terminal::

    pip install pyluna

To install luna subpackages, with more features specify a subpackage {common, radiology, pathology} or install all with ``pyluna[all]``.
For example, to install luna pathology, run this command::

    pip install pyluna[pathology]


This is the preferred method to install luna, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


3. Setup LUNA_HOME and Configurations
-------------------------------------

**LUNA_HOME** is an environment variable that points to a location where luna configs will be stored.

1. Set ``$LUNA_HOME`` as the current directory::

    export LUNA_HOME=<path/to/your/workspace>

2. Prepare configuration files and place them at ``$LUNA_HOME/conf`
Example configurations files are at `Luna repo <https://github.com/msk-mind/luna/tree/dev/conf>`_

Currently, we have two configuration files.

- **datastore.cfg**: configuration for the backend store of your data. POSIX and Minio backends are supported.
- **logging.cfg**: configuration for logging level and optional central logging in MongoDB.
