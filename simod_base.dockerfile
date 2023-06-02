FROM openjdk:8-jre-slim-bullseye

RUN apt-get update && apt-get install -y \
    git \
    glpk-utils \
    libblas-dev \
    libglpk-dev \
    liblapack-dev \
    libxext6 \
    libxrender1 \
    libxtst6 \
    python3 \
    python3-pip \
    python3-venv \
    python3-cvxopt \
    x11-utils \
    xvfb

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

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
