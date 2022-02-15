FROM python:3.9-buster

RUN apt-get update && \
    apt-get install -y openslide-tools python-openslide default-jre libblas-dev liblapack-dev libgdal-dev gfortran

ADD requirements_dev.txt .

# spams install error with 1.22.x
RUN pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install numpy==1.21.2 && \
    pip --no-cache-dir install -r requirements_dev.txt --use-deprecated=legacy-resolver

