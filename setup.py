#!/usr/bin/python3

from setuptools import setup

setup(
    name="fusion",
    packages=["fusion"],
    install_requires=["numpy"],
    extras_require={
        "testing": [
            "tinygrad==0.9.0",
            # "pytest",
            # "safetensors",
            # "transformers",
        ],
    },
)
