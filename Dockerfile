FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY ./data_processing /app/data_processing

RUN pip install --upgrade pip && pip --no-cache-dir install -r /app/data_processing/getPathologyAnnotations/requirements.txt 

COPY ./config.yaml.template /app/config.yaml

ENTRYPOINT ["python", "-m", "data_processing.getPathologyAnnotations.getPathologyAnnotations"]

