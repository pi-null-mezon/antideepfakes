API-deepfake-detection-tool
===

HTTP server for deepfake challenge 

Installation:

```bash
git clone https://github.com/pi-null-mezon/antideepfakes.git && cd antideepfakes
```

Build and save:

``` bash
cd .. && \
mkdir -p ./weights/final && \
...ask me to download weights and place them in ./weights/final...
docker build -f api/Dockerfile -t systemfailure/ddt . && \
docker save systemfailure/ddt | gzip > systemfailure-deepfake_detection_tool-v1.0.0.tar.gz
```

Run as standalone container:

```bash
docker run -p 8080:5000 -e PATH_PREFIX=face --gpus all systemfailure/ddt
```

Run as docker composition:

```bash
docker-compose up
```

Check api:

```
go to http://localhost:5000/docs
```