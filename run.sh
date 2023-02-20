#!/usr/bin/env bash

# This script is used for running Simod from a Docker container.
# It starts the virtual X11 server and runs Simod via poetry.

Xvfb :99 &>/dev/null & disown

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

# run simod
poetry run simod optimize --config_path $CONFIG_PATH --output_dir $OUTPUT_DIR
