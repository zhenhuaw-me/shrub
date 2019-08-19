#!/bin/sh

export PYTHONPATH=${PYTHONPATH}:$(dirname $(dirname $(readlink -f $0)))/python/
