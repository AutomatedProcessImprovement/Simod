#!/usr/bin/env bash

sonar-scanner \
  -Dsonar.projectKey=Simod \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.login=036c22343a718282417890a038dc8d374bbe8995
