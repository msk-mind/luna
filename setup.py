#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.md').read()
doclink = """
Documentation
-------------

The full documentation is at http://data-processing.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='data-processing',
    version='0.0.1',
    description='Scripts for data processing',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Doori Rose',
    author_email='rosed2@mskcc.org',
    url='https://github.com/doori/data-processing',
    packages=[
        'data_processing',
        'data_processing.common',
        'data_processing.services',
        'data_processing.scanManager',
        'data_processing.pathology.proxy_table',
        'data_processing.radiology.proxy_table',
        'data_processing.radiology.proxy_table.annotation',
        'data_processing.radiology.refined_table',
        'data_processing.radiology.refined_table.annotation',
        'data_processing.radiology.feature_table',
        'data_processing.radiology.feature_table.annotation'
    ],
    package_dir={'data-processing': 'data-processing'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='data-processing',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
