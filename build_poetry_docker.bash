#!/usr/bin/env bash

PROJECT_DIR=$(pwd)
CVXOPT_DIR=/usr/src/cvxopt

pip install -U pip
pip install poetry

poetry run pip install -U pip

cd $CVXOPT_DIR
poetry run pip install .

cd $PROJECT_DIR
poetry install
