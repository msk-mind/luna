#!/usr/bin/env bash

### find open ports and apply to the tutorial build and configuration files


# find and apply open port for girder
PORT=$(find_open_port.sh 8080 8180)

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:8080/$PORT:8080/g" docker-compose.yml
    sed -i '' -E "s/host.docker.internal:[0-9]{4}/host.docker.internal:$PORT/g" vmount/conf/dsa_regional_annotation.yaml
    sed -i '' -E "s/host.docker.internal:[0-9]{4}/host.docker.internal:$PORT/g" vmount/conf/dsa_point_annotation.yaml
else   # for linux
    sed -i '' -r "s/[0-9]{4}:8080/$PORT:8080/g" docker-compose.yml
    sed -i '' -r "s/host.docker.internal:[0-9]{4}/host.docker.internal:$PORT/g" vmount/conf/dsa_regional_annotation.yaml
    sed -i '' -r "s/host.docker.internal:[0-9]{4}/host.docker.internal:$PORT/g" vmount/conf/dsa_point_annotation.yaml
fi


# find and apply open port for jupyterlab
PORT=$(find_open_port.sh 8888 8988)

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:8888/$PORT:8888/g" docker-compose.yml
else   # for linux
    sed -i '' -r "s/[0-9]{4}:8888/$PORT:8888/g" docker-compose.yml
fi


PORT=$(find_open_port.sh 6006 6106)

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:6006/$PORT:6006/g" docker-compose.yml
else   # for linux
    sed -i '' -r "s/[0-9]{4}:6006/$PORT:6006/g" docker-compose.yml
fi