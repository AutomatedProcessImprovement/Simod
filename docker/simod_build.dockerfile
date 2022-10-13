FROM nokal/simod-base:v2.0.0

WORKDIR /usr/src/
ADD build.bash .
RUN bash build.bash

ENV DISPLAY=:99
CMD ["/bin/bash"]
