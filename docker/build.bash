#!/usr/bin/env bash

BRANCH_NAME="master"
BASE_DIR=/usr/src

# cloning repositories
cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git
cd Simod
git checkout $BRANCH_NAME
git submodule update --init --recursive

# creating virtual environment
python3 -m pip install --upgrade pip
python3 -m venv venv
source venv/bin/activate

# installing dependencies
cd $BASE_DIR/Simod/external_tools/Prosimos
pip install -e .
cd $BASE_DIR/Simod/external_tools/pm4py-wrapper
pip install -e .

# installing Simod
cd $BASE_DIR/Simod
pip install -e .
