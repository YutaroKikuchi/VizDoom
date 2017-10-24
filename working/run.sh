#!/bin/bash

if [ $# -ne 1 ] ; then
  echo usage : 1 parameter is required
  exit 1
fi

DIRECTORY=`basename $1`

docker run -ti --rm --name ${DIRECTORY} ${DIRECTORY}
