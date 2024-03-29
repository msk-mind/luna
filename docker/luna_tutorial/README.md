Luna Tutorial
=============
Prerequisites: 

Host OS - Mac or Linux.

Docker version 1.13+
Docker-Compose version 1.18+

See make help for instructions on how to build and run the tutorial docker container. 

```
$ make help [ARGS]

clean                cleanup luna-tutorial images and containers
clean-image          remove luna tutorial image
build                build containers
run                  launch containers
stop                 terminate containers but keep volumes
exec                 launches terminal prompt inside luna_tutorial container

```

You may then step through the notebooks in jupyterlab. Note that the url by default is set to localhost. If you need to run the tutorial from a remote system, you may specify the host IP and port of the remote system as arguments to the Makefile. 

**NOTE:** The tutorial image must be built from scratch by each user who wants to run the tutorial because the image is designed to mimic the host system user in the container in order to make it easy for the host system and container users to CRUD data in a shared volume mount (called vmount) from either the host system or the container. If an image built by one user is reused by another user, the other user may not be able to update or delete data generated from within the container without root permissions on the host system.  


