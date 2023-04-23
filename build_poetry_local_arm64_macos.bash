#!/usr/bin/env bash

PROJECT_DIR=$(pwd)

brew install glpk

cd $PROJECT_DIR
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git external_tools/SuiteSparse
cd external_tools/SuiteSparse && git checkout v5.6.0
SUITE_SPARSE_DIR=$(pwd)/external_tools/SuiteSparse

cd $PROJECT_DIR
git clone https://github.com/cvxopt/cvxopt.git external_tools/cvxopt
cd external_tools/cvxopt && git checkout $(git describe --abbrev=0 --tags)
CVXOPT_DIR=$(pwd)/external_tools/cvxopt

cd $PROJECT_DIR
pip install -U pip
pip install poetry
poetry run pip install -U pip

cd $PROJECT_DIR
CVXOPT_SUITESPARSE_SRC_DIR=$SUITE_SPARSE_DIR \
  CVXOPT_BUILD_GLPK=1 \
  CVXOPT_GLPK_LIB_DIR=/opt/homebrew/Cellar/glpk/5.0/lib \
  CVXOPT_GLPK_INC_DIR=/opt/homebrew/Cellar/glpk/5.0/include \
  poetry run pip install -e .

cd $CVXOPT_DIR
CVXOPT_SUITESPARSE_SRC_DIR=$SUITE_SPARSE_DIR \
  CVXOPT_BUILD_GLPK=1 \
  CVXOPT_GLPK_LIB_DIR=/opt/homebrew/Cellar/glpk/5.0/lib \
  CVXOPT_GLPK_INC_DIR=/opt/homebrew/Cellar/glpk/5.0/include \
  poetry run pip install -e .
