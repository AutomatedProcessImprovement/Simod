FROM nokal/simod-base:2.2.0

WORKDIR /usr/src/Simod
COPY . .
RUN pip install -U pip \
    && pip install poetry
RUN poetry install

ENV DISPLAY=:99
CMD ["/bin/bash"]

# Run "Xvfb :99 &>/dev/null & disown" before running Simod to start the X server for Java dependencies

# Docker usage example:
# $ docker run --rm -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash