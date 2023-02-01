#!/usr/bin/env bash

docker buildx build --platform linux/amd64,linux/arm64 -t nokal/simod -f Dockerfile --push .