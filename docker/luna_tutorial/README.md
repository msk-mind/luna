Luna Tutorial
=============
See make help for instructions on how to build and run the tutorial docker container. 

```
$ make help [ARGS]

clean                cleanup all images volumes and containers
clean-image          remove luna tutorial image
build                build luna tutorial docker image
run                  run the luna-tutorial container. Default IP:PORT=127.0.0.1:8888. Optionally specify args HOST_IP=xxx.xxx.xxx.xxx HOST_PORT=xxxx.
exec                 enter the luna-tutorial container

```

In order to run this tutorial, first build the docker image and then lauch the container. Once the container is lauched, go to the HOST_IP and PORT in your browser and paste in the token from the terminal into the brower token textbox. This should open the jupyterlab environment containing the tutorial notebooks. Then step through the notebooks. Note that the url by default is set to localhost. If you need to run the tutorial from a remote system, you may specify the host IP and port of the remote system as arguments to the Makefile. 

**NOTE:** The tutorial image must be built from scratch by each user who wants to run the tutorial because the image is designed to mimic the host system user in the container in order to make it easy for the host system and container users to CRUD data in a shared volume mount (called vmount) from either the host system or the container. If an image built by one user is reused by another user, the other user may not be able to update or delete data generated from within the container without root permissions on the host system.  


