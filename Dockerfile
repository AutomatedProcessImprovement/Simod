FROM eclipse-temurin:8-jdk-jammy

# Install prerequisites and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create and activate a Python 3.11 virtual environment
RUN python3.11 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /usr/src/Simod

COPY . .

# Install Poetry using Python 3.11
RUN python -m pip install --upgrade pip \
    && python -m pip install poetry

# Install Simod and its dependencies using Python 3.11
RUN poetry install

CMD ["/bin/bash"]

# Docker usage example:
# $ docker run --rm -it -v /path/to/resources/:/usr/src/Simod/resources -v /path/to/output:/usr/src/Simod/outputs nokal/simod bash
