FROM nokal/simod-base:v1.1.4

WORKDIR /usr/src/
ADD build.bash .
RUN bash build.bash

ENV DISPLAY=:99
CMD ["/bin/bash"]
