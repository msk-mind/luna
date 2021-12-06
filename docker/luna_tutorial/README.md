Luna Tutorial
=============
See make help for instructions on how to build and run the tutorial docker container. 

```
$ make help
```

In order to run this tutorial, first build the docker image and then lauch the container. Once the container is lauched, copy the url in the container's terminal log and paste it in a browser to open the jupyterlab environment containing the tutorial notebooks. Then step through the notebooks. Note that the url by default is set to localhost. If you need to run the tutorial from a remote system, replace the localhost IP address with the remote system IP address in the Makefile run target and rerun the container.  

**NOTE:** The tutorial image must be built from scratch by each user who wants to run the tutorial because the image is designed to mimic the host system user in the container in order to make it easy for the host system and container users to CRUD data in a shared volume mount (called vmount) from either the host system or the container. If an image built by one user is reused by another user, the other user may not be able to update or delete data generated from within the container without root permissions on the host system.  


