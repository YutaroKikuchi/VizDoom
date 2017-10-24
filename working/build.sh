
#!/bin/bash

if [ $# -ne 1 ] ; then
  echo "usage : 1 parameter is requested"
  exit 1
fi

DIRECTORY=`basename $1`
#echo ${DIRECTORY}
cd ${DIRECTORY}

docker build -t ${DIRECTORY} .

exit 0
