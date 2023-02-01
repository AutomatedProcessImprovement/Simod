FROM nokal/simod-base:v2.1.0

RUN apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/Simod
ADD . .
RUN bash build_poetry_docker.bash

RUN rm -rf /usr/src/SuiteSparse && rm -rf /usr/src/cvxopt

ENV DISPLAY=:99
CMD ["/bin/bash"]

# Run "Xvfb :99 &>/dev/null & disown" before running Simod to start the X server for Java dependencies

# Docker usage example:
# $ docker run --rm -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash