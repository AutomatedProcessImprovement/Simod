#!/usr/bin/env bash

# Builds and pushes the docker image to the docker hub for both platforms, amd64 and arm64.
docker buildx build --platform linux/amd64,linux/arm64 -t nokal/simod -f simod_build.dockerfile --push ./docker