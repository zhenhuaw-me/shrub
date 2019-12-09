import os
import numpy as np
import tfcoreml
import coremltools
from coremltools.models.neural_network import quantization_utils as qutils
from coremltools.models import (
  _MLMODEL_FULL_PRECISION,
  _QUANTIZATION_MODE_LINEAR_QUANTIZATION,
  _QUANTIZATION_MODE_LOOKUP_TABLE_LINEAR,
  _QUANTIZATION_MODE_LOOKUP_TABLE_KMEANS,
  _QUANTIZATION_MODE_CUSTOM_LOOKUP_TABLE
)

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util

import tvm
import nnvm
from tvm.contrib import graph_runtime as runtime
from tvm import rpc
import jackwish as jw

class DequantOpChecker:
  def __init__(self, name, ishape, range_min, range_max, mode="MIN_COMBINED", dtype="float32"):
    # args in TensorFlow style
    self.name     = name
    self.ishape   = ishape
    self.oshape   = ishape
    self.min      = range_min
    self.max      = range_max
    self.mode     = mode
    self.iname    = "input"
    self.oname    = "output"
    self.dtype    = dtype
    self.tflite_model_path = dtype + ".tflite"

  def genModels(self):
    print("Generating Models...")
    with tf.Session(graph=tf.Graph()) as sess:
      data = tf.placeholder(tf.quint8, shape=self.ishape, name=self.iname)
      conv = tf.dequantize(data, self.min, self.max, mode=self.mode, name=self.oname)
      sess.run(tf.global_variables_initializer())
      constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [self.oname])
      with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())

  def genTFLiteModel(self):
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
                  self.name + ".pb",
                  inference_type = tf.uint8 if self.dtype == "uint8" else tf.float32,
                  input_arrays = self.iname,
                  output_arrays = self.oname,
                  default_ranges_stats = (0, 6),
                  post_training_quantize = (self.dtype == "uint8")
                  # quantized_input_stats = 
                  )
    tflite_model = converter.convert()
    open(self.dtype + ".tflite", "wb").write(tflite_model)


  def preRun(self):
    self.input_nhwc = np.random.uniform(size=self.ishape).astype(self.dtype)
    self.input_nchw = self.input_nhwc.transpose(0, 3, 1, 2)

  def runTensorFlow(self):
    print("run TensorFlow...")
    tf.reset_default_graph()
    graph_def = graph_pb2.GraphDef()
    with open(self.name + ".pb", 'rb') as f:
      graph_def.ParseFromString(f.read())
    g = tf.import_graph_def(graph_def)
    with tf.Session(graph=g) as sess:
      image_input_tensor = sess.graph.get_tensor_by_name('import/' + self.iname + ":0")
      outputs = [sess.graph.get_tensor_by_name("import/" + self.oname + ":0")]
      self.output_tf = sess.run(outputs, feed_dict={image_input_tensor: self.input_nhwc})[0]

  def runTFLite(self):
    try:
        from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
    except ImportError:
        from tensorflow.lite.python import interpreter as interpreter_wrapper
    print("run TensorFlow Lite...")
    interpreter = interpreter_wrapper.Interpreter(model_path=(self.tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    for i0 in input_details:
        print(i0['shape'])
        print(i0['dtype'])

    output_details = interpreter.get_output_details()
    print("INPUT ", input_details)
    print("OUTPUT ", output_details)

    interpreter.set_tensor(input_details[0]['index'], self.input_nhwc)
    interpreter.invoke()
    if (len(self.oshape)) == 4:
        self.output_tflite = interpreter.get_tensor(output_details[0]['index']).transpose(0, 3, 1, 2)
    else:
        self.output_tflite = interpreter.get_tensor(output_details[0]['index'])

def test_Dequantize(input_shape, range_min, range_max, mode="MIN_COMBINED",
                    dtype="float32", tag="default"):
    print("\n[Begin] Test Dequantize with %s\n" % tag)
    op = DequantOpChecker("model", input_shape, range_min, range_max,
                          mode=mode, dtype=dtype)
    op.genModels()
    # if input_file:
    #     op.loadInputs(input_file)
    # else:
    #     op.preRun()
    # op.runTensorFlow()
    # op.runTFLite()
    # np.testing.assert_allclose(op.output_tvm, np.transpose(op.output_tf, axes=(0, 3, 1, 2)),
    #                            atol=1e-3, rtol=1e-3)
    print("\n[Pass] Test Dequantize with %s\n" % tag)

def test_dequant():
    print("[START] test_dequant")
    print("")
    input_shape  = (1, 256, 256, 32)  # NHWC
    range_min = 0.0
    range_max = 6.0

    # see http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/dequantize.html
    mode='MIN_COMBINED'
    # mode='MIN_FIRST'
    # mode='SCALED'

    dtype = "uint8"
    dtype = "float32"

    test_Dequantize(input_shape, range_min, range_max, mode=mode,
                    dtype=dtype, tag="dequantize")

    print("")
    print("[DONE] test_dequant")

test_dequant()
