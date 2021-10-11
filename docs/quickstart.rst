===========
Quick Start
===========

Luna docker images are available to help you get started with Luna.
Before running these images, be sure to install `Docker <https://docs.docker.com/get-docker/>`_ on your workspace.

These setup of notebook tutorials walk you through analysis workflows.
A static version of the :ref:`Tutorials` is also available on the documentation site for your reference.

Running Luna Pathology Tutorials on Docker
==========================================
We provide a docker to help you get started with pathology analysis using Luna workflows. This docker includes:

- Luna pathology library
- Jupyter notebook
- notebook tutorials with sample data

1. Clone the Git Repository
---------------------------
.. code-block::

    git clone https://github.com/msk-mind/docker.git


2. Run the container
--------------------
- Build the docker image

.. code-block::

    cd docker/luna_tutorial
    make build

- Start the jupyter (-d? why docker rm?)

.. code-block::

    make run

If you want to

.. code-block::

    make run



Running Digital Slide Archive (DSA) with sample images on Docker
================================================================

`Digital Slide Archive (DSA) <https://digitalslidearchive.github.io>`_ is used as a data management platform for organizing pathology images, creating annotations, and visualizing results.

This docker includes:

- a minimal DSA with girder app and mongo db
- a collection with sample images from CPTAC-OV and annotations created by non-experts for demo purposes only

1. Clone the git repository
---------------------------
.. code-block::

    git clone https://github.com/msk-mind/docker.git

2. Run the container
--------------------
.. code-block::

    cd docker/dsa_sample_data
    docker-compose up -d