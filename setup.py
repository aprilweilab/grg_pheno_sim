from setuptools import setup, find_packages
import os

PACKAGE_NAME = "grg_pheno_sim"
VERSION = "1.0"

THISDIR = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(THISDIR, "requirements.txt")) as f:
    requirements = list(map(str.strip, f))
with open(os.path.join(THISDIR, "README.md")) as f:
    long_description = f.read()

setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    version=VERSION,
    description="Phenotype simulator for GRGs",
    author="Aditya Syam",
    author_email="",
    url="https://aprilweilab.github.io/",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        requirements,
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
