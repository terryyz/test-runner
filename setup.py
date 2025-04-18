#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="test-runner",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to run tests from previously extracted test.jsonl files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/test-runner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "test-runner=run_tests:main",
            "run_tests=run_tests:main",
        ],
    },
) 