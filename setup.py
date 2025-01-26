#! /usr/bin/env python

import pathlib
from setuptools import setup, find_packages


__here__ = pathlib.Path(__file__).parent


def read(path):
    with open(path) as f:
        return f.read()


extra_files = {"WrightSim": ["VERSION"]}
version = read(__here__ / "WrightSim" / "VERSION").strip()


setup(
    name="WrightSim",
    packages=find_packages(),
    package_data=extra_files,
    python_requires=">=3.7",
    install_requires=[
        "h5py>=2.7.0",
        "numpy",
        "scipy",
        "WrightTools"
    ],
    extras_require={
        "docs": ["sphinx-gallery>=0.1.9"], 
        "cuda": ["pycuda"],
        "dev": [
            "black",
            "pre-commit",
            "pytest",
            "pytest-cov",
        ]
    },
    version=version,
    description="A simulation package for multidimensional spectroscopy.",
    long_description=read(__here__ / "README.rst"),
    long_description_content_type="text/x-rst",
    author="WrightSim Developers",
    license="MIT",
    url="https://github.com/wright-group/WrightSim",
    keywords="spectroscopy science multidimensional simulation",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
)
