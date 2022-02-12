"""
Allows installation via pip, e.g. by navigating to this directory with the command prompt, and using 'pip install .'
"""

import sys
from setuptools import setup, find_packages

setup(
    name='bamboo',
    version='0.2.1',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy'],
    description='Cooling system modelling for liquid rocket engines',
)
