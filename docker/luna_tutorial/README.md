Luna Tutorial
=============
See make help for instructions on how to build and run the tutorial docker container. 

```
$ make help [ARGS]

clean                cleanup luna-tutorial images and containers
clean-image          remove luna tutorial image
build                build containers
run                  launch containers. Default IP:PORT=127.0.0.1:8888. Optionally specify args HOST_IP=xxx.xxx.xxx.xxx HOST_PORT=xxxx.
stop                 terminate containers but keep volumes
exec                 launches terminal prompt inside luna_tutorial container

```

In order to run this tutorial, first build the docker-compose image and then lauch the containers. Once the containers are launched, you may view the various interfaces at these urls. 

    jupyterlab: http://localhost:8888  # NOTE insert token from vmount/logs/tutorial.log 
    DSA:        http://localhost:8080

You may then step through the notebooks in jupyterlab. Note that the url by default is set to localhost. If you need to run the tutorial from a remote system, you may specify the host IP and port of the remote system as arguments to the Makefile. 

**NOTE:** The tutorial image must be built from scratch by each user who wants to run the tutorial because the image is designed to mimic the host system user in the container in order to make it easy for the host system and container users to CRUD data in a shared volume mount (called vmount) from either the host system or the container. If an image built by one user is reused by another user, the other user may not be able to update or delete data generated from within the container without root permissions on the host system.  


