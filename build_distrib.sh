#!/bin/bash

docker build -f api/Dockerfile -t systemfailure/ddt . && \
docker save systemfailure/ddt | gzip > systemfailure-deepfake_detection_tool-v2.0.0.tar.gz && \
zip systemfailure-deepfake_detection_tool.zip \
systemfailure-deepfake_detection_tool-v1.0.0.tar.gz \
.env \
docker-compose.yaml \
HOW_TO_RUN.md \
scripts/*.py \
scripts/*.md \
scripts/*.txt && \
rm systemfailure-deepfake_detection_tool-v2.0.0.tar.gz