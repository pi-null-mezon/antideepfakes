SystemFailure© API-deepfake-detection-tool
===

HTTP server for deepfake challenge

input: video with face >> output: video liveness score

Supports inference on x86_64 CPUs and Nvidia CUDA GPUS 

## Installation:

```bash
unzip systemfailure-deepfake_detection_tool.zip && \
docker load -i systemfailure-deepfake_detection_tool-v1.0.0.tar.gz
```

## Run

As docker container:

```bash
docker run -p 8080:5000 -e PATH_PREFIX=face --gpus all systemfailure/ddt
```

As docker composition:

```bash
docker-compose up
```

### Options

Control server performance via environment variables:

 - WORKERS - number of http server instances per container (default: 1)
 - VIDEO_STROBE - video strobe for video decoding (default: 5)
 - MAX_SEQUENCES - how often deepfake detection should be performed (default: 4)  
 - DOUBLE_RES_CHECK - deepfake detection results will be refined on two input resolutions if enabled (default: False)
 - FACE_DETECTOR_INPUT_SIZE - bigger values allow to detect smaller faces (default: 150)

## API

```
go to http://localhost:5000/docs
```

### Scripts

Includes [scripts](./scripts/README.md) to process local directories: *./scripts*


*designed by Alex.A.Taranov, november 2023, SystemFailure©*