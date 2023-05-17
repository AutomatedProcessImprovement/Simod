FROM nokal/simod-base:v2.1.0 as base

RUN apt clean && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip

FROM base as builder

WORKDIR /usr/src/Simod
ADD . .
RUN pip install poetry
RUN poetry install

ENV DISPLAY=:99
CMD ["/bin/bash"]

# Run "Xvfb :99 &>/dev/null & disown" before running Simod to start the X server for Java dependencies

# Docker usage example:
# $ docker run --rm -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash