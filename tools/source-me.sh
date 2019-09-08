#!/bin/sh


if [ "$(uname -s)" = "Darwin" ]; then
  export PYTHONPATH=${PYTHONPATH}:$(dirname $(dirname $(greadlink -f $0)))/python/
else
  export PYTHONPATH=${PYTHONPATH}:$(dirname $(dirname $(readlink -f $0)))/python/
fi
