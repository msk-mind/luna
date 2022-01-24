FROM adoptopenjdk/openjdk16:x86_64-debian-jdk-16.0.1_9

RUN apt-get update && apt-get install -y sudo tree wget git binutils libgl1-mesa-glx libxml2-dev openslide-tools

# Checkout and build the main qupath library
RUN git clone https://github.com/qupath/qupath \
    && cd qupath \
    && git checkout tags/v0.3.2 -b v0.3.2 \
    && ./gradlew clean jpackage -Pcuda-redist

ENV PATH=/qupath/build/dist/QuPath/bin/:$PATH

# Checkout and build stardist extention
RUN git clone https://github.com/qupath/qupath-extension-stardist \
    && cd qupath-extension-stardist \
    && ./gradlew clean build \
    && cp /qupath-extension-stardist/build/libs/qupath-extension-stardist-0.3.0.jar /qupath/build/dist/QuPath/lib/app/

# Update Qupath.cfg
RUN sed -i '/Application/ a app.classpath=$APPDIR/qupath-extension-stardist-0.3.0.jar' /qupath/build/dist/QuPath/lib/app/QuPath.cfg &&  \
    sed -i '/JavaOptions/ a java-options=-Djava.awt.headless=true' /qupath/build/dist/QuPath/lib/app/QuPath.cfg

RUN cat /qupath/build/dist/QuPath/lib/app/QuPath.cfg

# Checkout models
RUN git clone https://github.com/qupath/models

# Copy scripts
COPY scripts /scripts/