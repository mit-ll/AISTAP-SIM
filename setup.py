# Copyright 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

from setuptools import find_packages, setup

setup(
    name="AISTAP_SIM",
    version="0.1.0",
    description="A package to load AISTAP_SIM datasets for ML",
    url="https://github.com/mit-ll/AISTAP-SIM",
    author="Michael Newey, Allan Wollaber, Dalton Vega",
    author_email="",
    license="MIT",
    packages=find_packages(where="src", exclude=["tests", "*.tests", "*.tests.*"]),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15",
        "numpy>=1.24,<2.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "tqdm>=4.66",
        "hydra-zen>=0.12.1",
        "natsort>=8.4",
        "pymatreader>=0.0.32",
        "h5py>=3.10",
        "pandas>=2.1",
        "pyyaml>=6.0"
    ],
    zip_safe=True,
)
