#!/usr/bin/env bash

BRANCH_NAME="master"
PROSIMOS_BRANCH_NAME="previous_json_format"
BASE_DIR=/usr/src
PROJECT_DIR=${BASE_DIR}/Simod
VENV_DIR=${PROJECT_DIR}/venv

# Cloning repositories
cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git $PROJECT_DIR
cd $PROJECT_DIR
git checkout $BRANCH_NAME
git submodule update --init --recursive
cd external_tools/Prosimos && git checkout $PROSIMOS_BRANCH_NAME && git pull && cd ../..

# Creating virtual environment
python3 -m venv venv
source $VENV_DIR/bin/activate
pip install --upgrade pip

# Installing dependencies
cd ${PROJECT_DIR}/external_tools/Prosimos
pip install -e .
cd ${PROJECT_DIR}/external_tools/pm4py-wrapper
pip install -e .

# Installing Simod
cd $PROJECT_DIR
pip install -e .
