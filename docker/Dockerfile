FROM python:3-buster

RUN apt-get update && apt-get install -y openslide-tools python-openslide default-jre

ADD requirements_dev.txt .

RUN pip install --upgrade pip && pip install numpy && pip install -r requirements_dev.txt -q

