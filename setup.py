#!/usr/bin/env python

from setuptools import setup

setup(
    name='viztricks',
    version='0.0.1',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Recipes and helper functions for plotting with Python',
    url='http://github.com/perimosocordiae/viztricks',
    license='MIT',
    packages=['viztricks'],
    install_requires=[
        'numpy >= 1.8',
        'matplotlib >= 1.3.1',
    ],
    extras_require=dict(
        scipy_deps=['scipy >= 0.14'],
        sklearn_deps=['scikit-learn >= 0.15']),
)
