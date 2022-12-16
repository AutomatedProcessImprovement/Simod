#!/usr/bin/env bash

BRANCH_NAME="master"
PROSIMOS_BRANCH_NAME="nov_2022"
BASE_DIR=/usr/src
PROJECT_DIR=${BASE_DIR}/Simod
VENV_DIR=${PROJECT_DIR}/venv

# Cloning repositories
cd $BASE_DIR
git clone https://github.com/AutomatedProcessImprovement/Simod.git $PROJECT_DIR
cd $PROJECT_DIR
git checkout $BRANCH_NAME
git submodule update --init --recursive
cd external_tools/Prosimos && git checkout $PROSIMOS_BRANCH_NAME && git pull

# Creating virtual environment
cd $PROJECT_DIR
python3.10 -m venv venv
source $VENV_DIR/bin/activate
pip3.10 install --upgrade pip

# Removing cvxopt and glpk, because they're installed on OS level
pip3.10 uninstall -y cvxopt glpk

# Installing dependencies
cd ${PROJECT_DIR}/external_tools/log-similarity-metrics
pip3.10 install -e .
pip3.10 install dtw-python
cd ${PROJECT_DIR}/external_tools/start-time-estimator
pip3.10 install -e .
cd ${PROJECT_DIR}/external_tools/extraneous-activity-delays
pip3.10 install -e .
cd ${PROJECT_DIR}/external_tools/Prosimos
pip3.10 install -e .

# Installing cvxopt from source. Pre-compiled binaries cause problems on ARM. cvxopt is required by pm4py-wrapper
cd $BASE_DIR
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
pushd SuiteSparse
git checkout v5.6.0
popd
export CVXOPT_SUITESPARSE_SRC_DIR=$(pwd)/SuiteSparse
git clone https://github.com/cvxopt/cvxopt.git
cd cvxopt
git checkout $(git describe --abbrev=0 --tags)
python setup.py install
pip3.10 install glpk

cd $PROJECT_DIR
pip3.10 install pm4py-wrapper

# Installing Simod
pip3.10 install -e .
