FROM openjdk:8-jre-bullseye

RUN apt update && apt upgrade -y
RUN apt install -y  \
    build-essential  \
    zlib1g-dev  \
    libncurses5-dev  \
    libgdbm-dev  \
    libnss3-dev  \
    libssl-dev  \
    libreadline-dev  \
    libffi-dev  \
    libsqlite3-dev  \
    wget  \
    libbz2-dev \
    vim  \
    git  \
    libxrender1  \
    libxext6  \
    libxtst6  \
    xvfb  \
    x11-utils

RUN cd /home
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
RUN tar -xf Python-3.10.0.tgz
RUN cd Python-3.10.0 && ./configure --enable-optimizations && make -j 4 && make altinstall

CMD /bin/bash
