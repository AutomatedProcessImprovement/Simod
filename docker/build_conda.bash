#!/usr/bin/env bash

BRANCH_NAME="master"
PROSIMOS_BRANCH_NAME="previous_json_format"
BASE_DIR=/usr/src

# Cloning repositories
cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git
cd Simod
git checkout $BRANCH_NAME
git submodule update --init --recursive
cd external_tools/Prosimos && git checkout $PROSIMOS_BRANCH_NAME && git pull && cd ../..

# Creating conda environment
conda create -y --name simod python=3.9
conda activate simod
python3 -m pip install --upgrade pip

# Installing dependencies
cd external_tools/pm4py-wrapper
pip install -e .
cd ../Prosimos
pip install -e .
cd ../..
conda install -y -c conda-forge click pandas numpy networkx matplotlib lxml xmltodict jellyfish scipy tqdm PyYAML hyperopt pytz pytest cvxopt

# Installing Simod
pip install -e .