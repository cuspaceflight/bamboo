"""
Allows installation via pip, e.g. by navigating to this directory with the command prompt, and using 'pip install .'
"""

import sys
from setuptools import setup, find_packages

setup(
    name='bamboo',
    version='0.1f',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'ambiance'],
    description='Cooling system modelling for liquid rocket engines',
)
