"""
setup.py

Type:       Python Package Setup Script
Author:     Will Brandon
Created:    June 23, 2023
Revised:    July 2, 2023

Builds and installs the pywbu package.
"""

import setuptools as setup


# Use setuptools to build and/or install the package.
setup.setup(
    name='deepartist',
    version='1.0.0',
    description='DeepArtist data model',
    url='git@github.com:will-brandon/py-packs.git',
    author='Will Brandon, Evan Suslovich, Harish Varadarajan',
    packages=setup.find_packages(),
    package_data={
        'deepartist': ['data', 'charts', 'model.pth']
    },
    include_package_data=True
)