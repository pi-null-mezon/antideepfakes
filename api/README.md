API-deepfake-detection-tool
===

HTTP server for deepfake challenge 

Installation:

```bash
git clone https://github.com/pi-null-mezon/antideepfakes.git && cd antideepfakes
```

Build and save:

``` bash
docker build -f Dockerfile -t systemfailure/ddt .
docker save systemfailure/ddt | gzip > systemfailure-deepfake_detection_tool-v1.0.0.tar.gz
```

Run:

```bash
docker run -p 8080:5000 --gpus all systemfailure/ddt
```

Check api:

```
go to http://localhost:5000/docs
```