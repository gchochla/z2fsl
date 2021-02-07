'''Setup script
Usage: pip install .
To install development dependencies too, run: pip install .[dev]
'''
from setuptools import setup, find_packages

setup(
    name='z2fsl',
    version='v1',
    packages=find_packages(),
    url = 'https://github.com/gchochla/z2fsl',
    author='Georgios Chochlakis',
    scripts=[],
    install_requires=[],
    extras_require={
        'dev': [
            'torch',
            'torchvision',
            'sklearn',
            'numpy'
        ],
    },
)
