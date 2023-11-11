#!/bin/bash

docker build -f api/Dockerfile -t systemfailure/ddt . && \
docker save systemfailure/ddt | gzip > systemfailure-deepfake_detection_tool-v1.0.0.tar.gz && \
zip systemfailure-deepfake_detection_tool.zip systemfailure-deepfake_detection_tool-v1.0.0.tar.gz .env docker-compose.yaml api/HOW_TO_RUN.md &&
rm systemfailure-deepfake_detection_tool-v1.0.0.tar.gz