import os
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util


class OpChecker:
    def __init__(self, op, ishape, range_min, range_max, mode="MIN_COMBINED"):
        # args in TensorFlow style
        self.op = op
        self.name = 'gen'
        self.ishape = ishape
        self.oshape = ishape
        self.min = range_min
        self.max = range_max
        self.mode = mode
        self.iname = 'input'
        self.oname = 'output'
        if op == tf.quantize:
            self.idtype = tf.float32
            self.odtype = tf.quint8
            self.tflite_model_path = "quant.tflite"
            self.tflite_dtype = tf.float32
        elif op == tf.dequantize:
            self.idtype = tf.quint8
            self.odtype = tf.float32
            self.tflite_model_path = "dequant.tflite"
            self.tflite_dtype = tf.float32
        else:
            raise ValueError("Unkown op")

    def genModels(self):
        print("Generating Models...")
        with tf.Session(graph=tf.Graph()) as sess:
            data = tf.placeholder(
                self.idtype,
                shape=self.ishape,
                name=self.iname)
            output = self.op(
                data,
                self.min,
                self.max,
                tf.quint8,
                mode=self.mode,
                name=self.oname)
            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [self.oname])
            with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def genTFLiteModel(self):
        print("Generating TensorFlow Lite model...")
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            self.name + ".pb",
            input_arrays=[self.iname],
            output_arrays=[self.oname, ])
        # converter.inference_type = self.tflite_dtype
        # converter.inference_inpute_type = self.tflite_dtype
        converter.default_ranges_stats = (0, 6)
        converter.quantized_input_stats = {self.iname: (100, 100.0)}

        # converter.post_training_quantize = True
        # converter.target_ops = set([OpsSet.TFLITE_BUILTINS])

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)

    def preRun(self):
        self.input_nhwc = np.random.uniform(
            size=self.ishape).astype(
            self.dtype)
        self.input_nchw = self.input_nhwc.transpose(0, 3, 1, 2)

    def runTensorFlow(self):
        print("run TensorFlow...")
        tf.reset_default_graph()
        graph_def = graph_pb2.GraphDef()
        with open(self.name + ".pb", 'rb') as f:
            graph_def.ParseFromString(f.read())
        g = tf.import_graph_def(graph_def)
        with tf.Session(graph=g) as sess:
            image_input_tensor = sess.graph.get_tensor_by_name(
                'import/' + self.iname + ":0")
            outputs = [
                sess.graph.get_tensor_by_name(
                    "import/" + self.oname + ":0")]
            self.output_tf = sess.run(
                outputs, feed_dict={
                    image_input_tensor: self.input_nhwc})[0]


def test_OP(op, input_shape, range_min, range_max, mode="MIN_COMBINED"):
    op = OpChecker(op, input_shape, range_min, range_max, mode=mode)
    op.genModels()
    op.genTFLiteModel()


def test_dequant():
    print("[START] test_dequant")
    print("")
    input_shape = (1, 256, 256, 32)  # NHWC
    range_min = 0.0
    range_max = 6.0

    # see
    # http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.google.cn/api_docs/python/tf/dequantize.html
    mode = 'MIN_COMBINED'
    # mode='MIN_FIRST'
    # mode='SCALED'

    quant_op = tf.dequantize
    quant_op = tf.quantize

    test_OP(quant_op, input_shape, range_min, range_max, mode=mode)

    print("")
    print("[DONE] test_dequant")


test_dequant()
