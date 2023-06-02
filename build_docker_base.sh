#!/usr/bin/env bash

# build the base image without using any build context
docker buildx build --platform linux/amd64,linux/arm64 -t nokal/simod-base:2.2.0 --push - < simod_base.dockerfile
