.. highlight:: shell

============
Installation
============

See below for installation instructions

1. Pre-requisites
-----------------

- `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_:
  Follow in the instructions for installing `Mambaforge
  <https://github.com/conda-forge/miniforge#mambaforge>`_
- `GNU Make <https://www.gnu.org/software/make/>`_:
  Version > 3.81 required. (Default version on macos is 3.81)

2. Create luna virtual environment and install packages
-------------------------------------------------------

At the command line::

    make venv

Activate the luna virtual environment::

    conda activate .venv/luna


3. Setup LUNA_HOME and Configurations
-------------------------------------

**LUNA_HOME** is an environment variable that points to a location where luna configs will be stored.

1. Set ``$LUNA_HOME`` as the current directory::

    export LUNA_HOME=<path/to/your/workspace>

2. Prepare configuration files and place them at ``$LUNA_HOME/conf``
Example configurations files are at `Luna repo <https://github.com/msk-mind/luna/tree/dev/conf>`_

Currently, we have two configuration files.

- **datastore.cfg**: configuration for the backend store of your data. POSIX and Minio backends are supported.
- **logging.cfg**: configuration for logging level and optional central logging in MongoDB.
