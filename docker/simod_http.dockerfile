FROM nokal/simod-base:v2.0.0

RUN apt update && apt install -y  \
    libblas-dev  \
    liblapack-dev \
    python3-cvxopt \
    glpk-utils  \
    libglpk-dev

WORKDIR /usr/src/
ADD build_from_git.bash .
RUN bash build_from_git.bash

ENV DISPLAY=:99
ENV SIMOD_HTTP_DEBUG=false
ENV VIRTUAL_ENV=/usr/src/simod/venv

WORKDIR /usr/src/Simod/src/simod_http

CMD ["/usr/src/Simod/venv/bin/uvicorn", "simod_http.main:app", "--host", "0.0.0.0", "--port", "80"]
