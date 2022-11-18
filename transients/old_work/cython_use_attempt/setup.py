from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("_transientsv3.pyx")
)
