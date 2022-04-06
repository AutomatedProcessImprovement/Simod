FROM nokal/simod-base:v1.1.4

RUN apt-get update && apt-get install -y python3-venv

WORKDIR /usr/src/
ADD build.bash .
RUN bash build.bash

ENV DISPLAY=:99
CMD ["/bin/bash"]
