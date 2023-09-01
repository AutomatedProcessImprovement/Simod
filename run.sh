#!/usr/bin/env bash

# This script is used for running Simod from a Docker container.

# configuration path from the command line
CONFIG_PATH=$1

# optional output_dir from the command line
OUTPUT_DIR=$2

# if no config_path is specified, exit with error
if [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: No configuration file specified."
    exit 1
fi

# if no output_dir is specified, use the default directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=$(pwd)/outputs
fi

# run Simod
poetry run simod --configuration $CONFIG_PATH --output $OUTPUT_DIR
