# -*- coding: utf-8 -*-
"""
@author: Duvérier DJIFACK ZEBAZE
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scientistshiny", 
    version="0.0.1",
    author="Duverier DJIFACK ZEBAZE",
    author_email="duverierdjifack@gmail.com",
    description="Python module for measure the degree of association between variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enfantbenidedieu/scientistmetrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)