#!/bin/bash
set -e

# Create a conda environment
conda create -y --name knmiradar python=3.10

# Activate the conda environment
call conda activate knmiradar

# Install packages from requirements_dev.txt and requirements.txt
pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks to check validity of code changes
pre-commit install
