#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

#absolute_path = os.path.abspath(relative_path)
cur_dir = os.getcwd()

file_dir = os.path.dirname(os.path.abspath(__file__))
pybind_file = os.path.join(file_dir, "pybind_calculate_local_energy.cpp")
include_path=[os.path.join(cur_dir, "include")]

BACKEND = os.getenv('__BACKEND')
PSI_DTYPE = os.getenv('__PSI_DTYPE')
ext_modules = [
    Pybind11Extension(
        "calculate_local_energy",
        [pybind_file],
        include_dirs=include_path,
        library_dirs=[cur_dir],
        libraries=['eloc'],
        extra_compile_args=[f'-D{BACKEND}', f'-D{PSI_DTYPE}'],
        extra_link_args=[f"-Wl,-rpath,{cur_dir}"],
    ),
]

setup(
    name="calculate_local_energy",
    version="0.0.1",
    author="Yangjun Wu",
    description="A pybind11 wrapper for calculate_local_energy C++ functions",
    # packages=find_packages(where="."),
    ext_modules=ext_modules,
    # py_modules=["eloc"],
    py_modules=["interface.python.eloc"],
    cmdclass={"build_ext": build_ext},
)

# python interface/python/setup.py build_ext --inplace
# python setup.py build_ext --inplace
