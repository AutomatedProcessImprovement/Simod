#!/usr/bin/env bash

PROJECT_DIR=$(pwd)
CVXOPT_DIR=/usr/src/cvxopt

pip install -U pip
pip install poetry

poetry shell
pip install -U pip

cd $CVXOPT_DIR
pip install .

cd $PROJECT_DIR
poetry install
