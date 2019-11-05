import setuptools
from setuptools import setup


__version__ = "0.0.1"
__description__ = (
    "Some basic utilities for working with PyTorch"
    " and torms3's DataProvider for connectomics"
)


setup(
    name="pytorchutils",
    version=__version__,
    description=__description__,
    author="Nicholas Turner",
    author_email="nturner@cs.princeton.edu",
    url="https://github.com/nicholasturner1/PyTorchUtils",
    packages=setuptools.find_packages()
)
