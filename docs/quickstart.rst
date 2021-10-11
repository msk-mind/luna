===========
Quick Start
===========

Luna docker images are available to help you get started with Luna.
Before running these images, be sure to install `Docker <https://docs.docker.com/get-docker/>`_ on your workspace.

Running Luna Pathology Tutorials on Docker
==========================================
We provide a docker image to help you get started with pathology analysis using Luna workflows.
This image includes:

- Luna pathology library
- JupyterLab
- notebook tutorials with sample data

These set of notebook tutorials walk you through analysis workflows.
A static version of the notebook :ref:`Tutorials` is also available on the documentation site for your reference.

1. Clone the git repository
---------------------------
.. code-block::

    git clone https://github.com/msk-mind/docker.git


2. Run the container
--------------------
- Build the docker image

.. code-block::

    cd docker/luna_tutorial
    make build

- Run the docker image

.. code-block::

    make run

Copy paste the Jupyter URL printed in the console (alternatively, open it from docker dashboard),
and you will see a set of ipynb notebooks for exploring end-to-end Luna pathology analysis workflow.


If you want to save any changes in the notebooks you've made to your local workspace,
run this command to save the notebooks in notebooks directory.

.. code-block::

    make backup



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

Go to `localhost:8080` and login with admin/password.
After login, you can navigate to the TCGA Collection, and explore images on HistomicsUI.
For details on using DSA, please refer to `DSA documentation <https://digitalslidearchive.github.io/digital_slide_archive/>`_
