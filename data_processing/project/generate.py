'''
Created on April 20, 2021

@author: rosed2@mskcc.org

This module generates a project directory on the data lake and copies a manifest yaml file
'''
import click
import os, shutil

from data_processing.common.CodeTimer import CodeTimer
from data_processing.common.config import ConfigSet
from data_processing.common.custom_logger import init_logger
import data_processing.common.constants as const

logger = init_logger()


@click.command()
@click.option('-m', '--manifest_file', default=None, type=click.Path(exists=True),
              help="path to yaml file containing project details. "
                   "See ./manifest.yaml.template")
def cli(manifest_file):
    """
    This module generates a project directory on the data lake and copies a manifest yaml file.

    Example:
        python3 -m data_processing.project.generate \
                 --manifest_file <path to manifest file>
    """
    with CodeTimer(logger, 'generate project folder'):
        cfg = ConfigSet(name=const.DATA_CFG, config_file=manifest_file)

        # create project dir and copy manifest yaml
        project_location = const.PROJECT_LOCATION(cfg)
        os.makedirs(project_location, exist_ok=True)

        shutil.copy(manifest_file, os.path.join(project_location, "manifest.yaml"))
        logger.info("config files copied to %s", project_location)



if __name__ == "__main__":
    cli()