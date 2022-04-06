FROM nokal/simod-base:v1.1.5

WORKDIR /usr/src/
ADD build.bash .
RUN bash build.bash

ENV DISPLAY=:99
CMD ["/bin/bash"]
