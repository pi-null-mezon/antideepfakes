FROM nvidia/cuda:11.4.0-runtime

EXPOSE 5000

LABEL Description="Deepfake detection server" \
      Vendor="SystemFailure" \
      Maintainer="pi-null-mezon@yandex.ru"

WORKDIR /usr/local/bin

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Enable unicode support
ENV LANG en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

COPY / /usr/local/bin
COPY ../weights/final /usr/local/weights/final

# Install dependencies
RUN apt-get update && \
    apt-get install python3 python3-pip libglib2.0-0 libgl1-mesa-dev -y && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /usr/local/bin/requirements.txt

ENTRYPOINT ["python3", "/usr/local/bin/httpsrv.py"]