#!/bin/env python

import os
import logging
import shrub

logging.basicConfig(level=logging.INFO)

################# XXX update these env accordingly XXX #############

target_name = 'cpu'
network = 'mobilenet'
useLayout = 'NCHW'
# useLayout = 'NHWC'
dtype = 'float32'

useCores = 4
bigLITTLE = 1

####################################################################

target = shrub.tvm.TargetProvider(target_name, bigLITTLE, useCores)
net, params, model = shrub.tvm.get_model(network, dtype, useLayout)

tuner = shrub.tvm.Tuner(model.name, target)
tuner.tune(net, params, target)
dep = shrub.tvm.Deployable()
dep.build(target, model, net, params, tuner.record)
dep.export(target)

model.genInput()
model.store('input')
rti = shrub.tvm.RuntimeWrapper(model, target, dep)
rti.run(model, useLayout, dumpOutput=True)
# shrub.tflite.run(model, dumpOutput=True, logLevel=logging.INFO)
# rti.profile(target, 10)
