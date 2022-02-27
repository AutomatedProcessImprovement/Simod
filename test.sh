#!/usr/bin/env bash

IMAGE_NAME="nokal/simod-testing:v1.3.0"
CONTAINER_NAME="simod_testing"

docker run --rm --name $CONTAINER_NAME $IMAGE_NAME bash test_simod.sh
