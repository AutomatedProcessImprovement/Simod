#!/usr/bin/env bash

BRANCH_NAME="master"
BASE_DIR=/usr/src

# cloning repositories
cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git
cd Simod
git checkout $BRANCH_NAME
git submodule update --init --recursive

# creating conda environment
conda create -y --name simod python=3.9
conda activate simod
python3 -m pip install --upgrade pip

# installing dependencies
cd external_tools/pm4py-wrapper
pip install -e .
cd ../Prosimos
pip install -e .
cd ../..
conda install -y -c conda-forge click pandas numpy networkx matplotlib lxml xmltodict jellyfish scipy tqdm PyYAML hyperopt pytz pytest cvxopt

# installing Simod
pip install -e .