#!/bin/bash

if [ $# -ne 1 ] ; then
  echo usage : 1 parameter is required
  exit 1
fi

if [[ ! -z  `which nvidia-docker`  ]]
then
    DOCKER_CMD=nvidia-docker
elif [[ ! -z  `which docker`  ]]
then
    echo "WARNING: nvidia-docker not found. Nvidia drivers may not work." >&2
    DOCKER_CMD=docker
else
     echo "${ERROR_PREFIX} docker or nvidia-docker not found. Aborting." >&2
    exit 1
fi

DIRECTORY=`basename $1`

if [ -e temp ]; then
  rm temp -rf
fi

$DOCKER_CMD run -ti --net=host --privileged -e DISPLAY=${DISPLAY} --name ${DIRECTORY} ${DIRECTORY}
$DOCKER_CMD cp ${DIRECTORY}:/home/vizdoom/ ./temp/
$DOCKER_CMD rm ${DIRECTORY}

