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
- Luna notebook tutorials with sample data

Along with web applications:

- JupyterLab for running notebooks
- Digital Slide Archive (girder and mongodb) for viewing the slides and annotations


These set of notebook tutorials walk you through analysis workflows.
A static version of the notebook :ref:`Tutorials` is also available on the documentation site for your reference.

1. Clone the git repository
---------------------------
.. code-block::

    git clone https://github.com/msk-mind/docker.git


2. Run the docker-compose
-------------------------
- Build the docker image

.. code-block::

    cd docker/luna_tutorial
    make build

- Run the docker image

.. code-block::

    make run

Copy paste the Jupyter URL printed in `luna_tutorial/vmount/logs/tutorial.log`,
and you will see a set of ipynb notebooks for exploring end-to-end Luna pathology analysis workflow.

To login to DSA, go to http://localhost:8080 and login with admin/password.
After login, you can navigate to the TCGA Collection, and explore images on HistomicsUI.
For details on using DSA, please refer to `DSA documentation <https://digitalslidearchive.github.io/digital_slide_archive/>`_

