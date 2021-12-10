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

On Ubuntu, you could install the pre-reqs with:

.. code-block:: console

    $ apt-get install -y openslide-tools python-openslide default-jre


2. Clone Git Repo
-----------------

Clone the public repository:

.. code-block:: console

    $ git clone git://github.com/msk-mind/luna

3. Install Python Dependencies and Luna Packages
------------------------------------------------

First, update your pip and install numpy:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install numpy # pyradiomics installation fails if numpy is not installed

.. note ::

    for pyluna-* packages that are not on pypi, add your local path in setup.cfg for installation to work correctly.

    For example, modify `absolute-path-to-luna-repo` and update your setup.cfgs with:

    pyluna-radiology @ file://localhost/absolute-path-to-luna-repo/pyluna-radiology/

To install luna and its basic functionality, run this command in your terminal:

.. code-block:: console

    $ pip install .

To install luna subpackages, with more features specify a subpackage {radiology, pathology} or install all with ``.[all]`` to installl all features.
For example, to install luna pathology, run this command:

.. code-block:: console

    $ pip install .[pathology]

Check that your installation was successful in Python shell.

.. code-block:: python

    >> import luna


3.1 Development Mode
********************

For development mode, install [dev] extras that includes dependencies for testing the code and generating documentation.

.. code-block:: console

    $ pip install .[dev]

Alternatively, you can install python dependencies from the ``requirements_dev.txt``
and setup ``$PYTHONPATH``

.. code-block:: console

    $ pip install -r requirements_dev.txt


3.2 Development with Docker
***************************

`luna-dev <https://hub.docker.com/r/doori87/luna-dev>`_ docker image is available on DockerHub.

This docker image includes the pre-requisites and python dependencies.
This is primarily used for circleci testing at the moment, but can be extended based on your development needs.

Once we have a stable release, docker images that includes pyluna packages can be made available.


4. Setup PYTHONPATH and LUNA_HOME
---------------------------------

In luna repo, run:

.. code-block:: console

    $ . setup_local.sh

Running the setup_local script will:

1. Add local luna packages to ``$PYTHONPATH``, so you can import luna packages in dev mode.

2. Set ``$LUNA_HOME`` as the current directory.
   Steps to setup configuration files in ``$LUNA_HOME/conf`` are detailed in the :ref:`Configuration section<5. Setup Configuration>`.



5. Setup Configuration
----------------------

Currently, we have two configuration files.

- **datastore.cfg**: configuration for the backend store of your data. POSIX and Minio backends are supported.

- **logging.cfg**: configuration for logging level and optional central logging in MongoDB.

In `$LUNA_HOME/conf/` folder,
copy ``logging.default.yml`` to ``logging.cfg`` and ``datastore.default.yml`` to ``datastore.cfg`` and modify the ``.cfg`` files.

.. code-block:: console

    $ cd $LUNA_HOME/conf
    $ cp logging.default.yml logging.cfg
    $ cp datastore.default.yml datastore.cfg


[Future] Stable Release
-----------------------

To install luna and its basic functionality, run this command in your terminal:

.. code-block:: console

    $ pip install pyluna

To install luna subpackages, with more features specify a subpackage {common, radiology, pathology} or install all with ``pyluna[all]``.
For example, to install luna pathology, run this command:

.. code-block:: console

    $ pip install pyluna[pathology]


This is the preferred method to install luna, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


JupyterLab Installation Tutorial
--------------------------------

This notebook tutorial will go help you setup your luna virtual environment in Jupyterlab.

.. toctree::
    :maxdepth: 1

    examples/setup.ipynb
