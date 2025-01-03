from setuptools import setup, find_packages

PACKAGE_NAME = "grg_pheno_sim"
VERSION = "1.0"

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
)