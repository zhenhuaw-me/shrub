#!/bin/bash

root_dir=$(dirname $(dirname $(readlink -f $0})))
doc_dir=${root_dir}/docs
proj_dir=${root_dir}/shrub

rm -rf ${doc_dir}
pdoc --overwrite --html --html-dir ${doc_dir} ${proj_dir} shrub
mv ${doc_dir}/shrub/* ${doc_dir}
rmdir ${doc_dir}/shrub

