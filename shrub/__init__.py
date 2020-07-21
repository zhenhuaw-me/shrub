from shrub import network
from shrub import onnx
from shrub import testing
from shrub import tflite
from shrub.tflite import TFLiteRunner
from shrub import util


# package metadata
NAME = 'shrub'
VERSION = '0.0.2-post1'
DESCRIPTION = "Toys to play around with machine learning"
LICENSE = 'Apache License Version 2.0'

__all__ = [
    network,
    onnx,
    testing,
    tflite,
    util,
    NAME,
    VERSION,
    DESCRIPTION,
    LICENSE,
]
