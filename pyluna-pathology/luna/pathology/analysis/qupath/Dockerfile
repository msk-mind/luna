FROM adoptopenjdk/openjdk16:x86_64-debian-jdk-16.0.1_9

RUN apt-get update && apt-get install -y sudo tree wget git binutils

RUN git clone https://github.com/qupath/qupath && cd qupath && git checkout tags/v0.3.2 -b v0.3.2

# RUN cd qupath && ./gradlew clean jpackage -Pcuda-redist
RUN cd qupath && ./gradlew clean jpackage

RUN git clone https://github.com/qupath/qupath-extension-stardist

RUN cd qupath-extension-stardist && ./gradlew clean build

RUN git clone https://github.com/qupath/models

RUN cp /qupath-extension-stardist/build/libs/qupath-extension-stardist-0.3.0.jar /qupath/build/dist/QuPath/lib/app/

RUN ls /qupath/build/dist/QuPath/bin/

RUN sed -i '10 a app.classpath=$APPDIR/qupath-extension-stardist-0.3.0.jar' /qupath/build/dist/QuPath/lib/app/QuPath.cfg

RUN sed -i '$ a java-options=-Djava.awt.headless=true' /qupath/build/dist/QuPath/lib/app/QuPath.cfg

RUN cat  /qupath/build/dist/QuPath/lib/app/QuPath.cfg

RUN apt-get install -y libgl1-mesa-glx  libxml2-dev openslide-tools

COPY scripts /scripts