from . import network
from . import onnx
from . import testing
from . import tflite
# from . import tvm
from . import util

# package metadata
NAME = 'shrub'
VERSION = '0.0.2'
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
