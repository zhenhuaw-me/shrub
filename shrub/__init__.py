from shrub import network
from shrub import onnx
from shrub import tflite
from shrub import mapping
from shrub.tflite import TFLiteRunner
from shrub.onnx import ONNXRunner
from shrub.predict import Classifier
from shrub import util


# package metadata
NAME = 'shrub'
VERSION = '0.0.2.post5'
DESCRIPTION = "Toys to play around with machine learning"
LICENSE = 'Apache License Version 2.0'

__all__ = [
    network,
    onnx,
    tflite,
    mapping,
    util,
    Classifier,
    TFLiteRunner,
    ONNXRunner,
    NAME,
    VERSION,
    DESCRIPTION,
    LICENSE,
]
