#!/usr/bin/env bash

### find open ports and apply to the tutorial build and configuration files


# find and apply open port for girder
echo "Finding available port for girder..."
PORT=$(./find_open_port.sh 8080 65535)
echo "Found port $PORT"
rm -f .dsa_port
echo $PORT >> .dsa_port

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:8080/$PORT:8080/g" docker-compose.yml
    sed -i '' -E "s/girder:[0-9]{4}/girder:$PORT/g" vmount/conf/dsa_regional_annotation.yaml
    sed -i '' -E "s/girder:[0-9]{4}/girder:$PORT/g" vmount/conf/dsa_point_annotation.yaml
    sed -i '' -E "s/port: [0-9]{4}/port: $PORT/g" vmount/conf/visualize_tiles.yaml
else   # for linux
    sed -i -r "s/[0-9]{4}:8080/$PORT:8080/g" docker-compose.yml
    sed -i -r "s/girder:[0-9]{4}/girder:$PORT/g" vmount/conf/dsa_regional_annotation.yaml
    sed -i -r "s/girder:[0-9]{4}/girder:$PORT/g" vmount/conf/dsa_point_annotation.yaml
    sed -i -r "s/port: [0-9]{4}/port: $PORT/g" vmount/conf/visualize_tiles.yaml
fi


# find and apply open port for jupyterlab
echo "Finding available ports for jupyterlab..."
PORT=$(./find_open_port.sh 8888 65535)
echo "Found port $PORT"
rm -f .jupyter_port
echo $PORT >> .jupyter_port

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:8888/$PORT:8888/g" docker-compose.yml
else   # for linux
    sed -i -r "s/[0-9]{4}:8888/$PORT:8888/g" docker-compose.yml
fi


PORT=$(./find_open_port.sh 6006 65535)
echo "Found port $PORT"

unameOut="$(uname -s)"

if [[ "$unameOut" == *"Darwin"* ]];
then   # for mac
    sed -i '' -E "s/[0-9]{4}:6006/$PORT:6006/g" docker-compose.yml
else   # for linux
    sed -i -r "s/[0-9]{4}:6006/$PORT:6006/g" docker-compose.yml
fi
