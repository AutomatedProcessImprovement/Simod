FROM openjdk:8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

WORKDIR /usr/src/Simod
COPY . .
RUN pip install -U pip
RUN pip install poetry
RUN poetry install

CMD ["/bin/bash"]

# Docker usage example:
# $ docker run --rm -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash