FROM python:3.9-buster

RUN apt-get update && \
    apt-get install -y openslide-tools python-openslide default-jre

ADD requirements_dev.txt .

RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install numpy && \
    pip --no-cache-dir install -r requirements_dev.txt -q

