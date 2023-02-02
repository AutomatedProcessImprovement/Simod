#!/usr/bin/env bash

Xvfb :99 &>/dev/null & disown

cd /usr/src/Simod/src/simod_http
export VIRTUAL_ENV=$(poetry env info --path)
poetry run uvicorn simod_http.main:app --host 0.0.0.0 --port 80