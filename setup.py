#!/usr/bin/env python

from setuptools import setup

setup(
    name='viztricks',
    version='0.1.0',
    author='CJ Carey',
    author_email='perimosocordiae@gmail.com',
    description='Recipes and helper functions for plotting with Python',
    url='http://github.com/perimosocordiae/viztricks',
    license='MIT',
    packages=['viztricks'],
    install_requires=[
        'numpy >= 1.6.1',
        'matplotlib >= 1.3.1',
    ],
    extras_require=dict(
        pca_ellipse=['scikit-learn >= 0.14'],
        voronoi_filled=['scipy >= 0.10'],
    ),
)
