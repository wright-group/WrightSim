#! /usr/bin/env python

import pathlib
from setuptools import setup, find_packages

__here__ = pathlib.Path(__file__).parent

extra_files = {"WrightSim": ["VERSION"]}

with __here__ / "WrightSim" / "VERSION" as version_file:
    version = version_file.read_text().strip()

setup(
    name="WrightSim",
    packages=find_packages(),
    package_data=extra_files,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=["h5py>=2.7.0", "numpy", "scipy", "WrightTools"],
    extras_require={"docs": ["sphinx-gallery>=0.1.9"], "cuda": ["pycuda"]},
    version=version,
    description="A simulation package for multidimensional spectroscopy.",
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
)
