FROM openjdk:8-jre-slim-bullseye

RUN apt update
RUN apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    libxrender1 \
    libxext6 \
    libxtst6 \
    xvfb \
    x11-utils \
    vim \
    git \
    libblas-dev \
    liblapack-dev \
    libglpk-dev \
    python3-cvxopt \
    glpk-utils \
    libglpk-dev

# pm4py requires cvxopt with glpk, for arm64 it needs to be compiled from source
WORKDIR /usr/src/
RUN git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
WORKDIR /usr/src/SuiteSparse
RUN git checkout v5.6.0
WORKDIR /usr/src/
RUN git clone https://github.com/cvxopt/cvxopt.git
WORKDIR /usr/src/cvxopt
RUN git checkout $(git describe --abbrev=0 --tags)
ENV CVXOPT_SUITESPARSE_SRC_DIR=/usr/src/SuiteSparse
ENV CVXOPT_BUILD_GLPK=1
WORKDIR /usr/src

CMD /bin/bash
