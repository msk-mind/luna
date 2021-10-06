#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

__version__ = '0.0.1'

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

# load setup.cfg. Preferring static setup.cfg to dynamic setup.py for deterministic lib creation
setup()
