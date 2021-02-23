FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN apt update 

RUN apt install default-jre -y

RUN export PYSPARK_PYTHON=`which python`

COPY ./data_processing /app/data_processing

RUN ls /app/data_processing

RUN pip install --upgrade pip && pip --no-cache-dir install -r /app/data_processing/get_pathology_annotations/requirements.txt 

RUN cp /app/data_processing/get_pathology_annotations/app_config.yaml.template /app/app_config.yaml

ENTRYPOINT ["python", "-m", "data_processing.get_pathology_annotations.get_pathology_annotations", "-c", "/app/app_config.yaml"]


