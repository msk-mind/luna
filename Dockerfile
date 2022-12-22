FROM mambaorg/micromamba:1.0.0

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
             PYTHONFAULTHANDLER=1 \
             PYTHONUNBUFFERED=1 \
             PYTHONHASHSEED=random \
             PIP_NO_CACHE_DIR=off \
             PIP_DISABLE_PIP_VERSION_CHECK=on \
             PIP_DEFAULT_TIMEOUT=100 \
             POETRY_VERSION=1.1.15

USER root
RUN apt-get update && apt-get install build-essential libgdal-dev liblapack-dev libblas-dev gfortran libgl1 -y

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml poetry.lock pyproject.toml /code/
WORKDIR /code
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)


RUN poetry export --without-hashes -f requirements.txt $(test "$YOUR_ENV" != production && echo "--dev")  | SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-deps -r /dev/stdin

COPY --chown=$MAMBA_USER:$MAMBA_USER . /code

RUN poetry config virtualenvs.create false && \
        poetry build &&  pip install --no-deps dist/*.whl
