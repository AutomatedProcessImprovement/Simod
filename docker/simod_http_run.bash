#!/usr/bin/env bash

Xvfb :99 &>/dev/null & disown

cd /usr/src/Simod
source venv/bin/activate

cd /usr/src/Simod/src/simod_http
uvicorn simod_http.main:app --host 0.0.0.0 --port 80