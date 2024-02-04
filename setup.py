#!/usr/bin/env python3

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ccm",
    version="0.0.0",
    author="Sadamori Kojaku",
    license="MIT",
    author_email="skojaku@binghamton.edu",
    description="Community Citation Models",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/skojaku/community_citation_models",
    packages=["iteremb"],
    install_requires=[line.strip() for line in open("requirements.txt")],
    python_requires=">=3.9",
)
