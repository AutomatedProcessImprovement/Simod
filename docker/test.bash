#!/usr/bin/env bash

# This script is run inside the container.

Xvfb :99 &>/dev/null & disown
cd /usr/src/Simod
source venv/bin/activate
pytest --exitfirst