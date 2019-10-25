#!/bin/bash

root_dir=$(dirname $(dirname $(readlink -f $0})))
rm -f ${root_dir}/assets/dist/bush-*.whl

python3 ${root_dir}/setup.py bdist_wheel \
  --bdist-dir ${root_dir}/assets/build \
  --dist-dir ${root_dir}/assets/dist
rm -rf ${root_dir}/bush.egg-info
rm -rf ${root_dir}/build
