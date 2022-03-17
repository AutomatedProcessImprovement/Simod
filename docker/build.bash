#!/usr/bin/env bash

BRANCH_NAME="master"
BASE_DIR=/usr/src

cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git simod
cd simod
git checkout $BRANCH_NAME
git submodule update --init --recursive
python3 -m pip install --upgrade pip

cd $BASE_DIR/simod/external_tools/Prosimos
pip install -e .

cd $BASE_DIR/simod/external_tools/pm4py-wrapper
pip install -e .

cd $BASE_DIR/simod
pip install -e .
