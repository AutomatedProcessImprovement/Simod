FROM openjdk:8-jre-bullseye

RUN apt update
RUN apt install -y python3 python3-pip python3-venv vim git libxrender1 libxext6 libxtst6 xvfb x11-utils

CMD /bin/bash
