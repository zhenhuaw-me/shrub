import os
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util


class OpChecker:
    def __init__(self, ishape, oshape):
        # args in TensorFlow style
        self.name = 'reshape'
        self.ishape = ishape
        self.oshape = oshape
        self.iname = 'input'
        self.oname = 'output'
        self.dtype = tf.float32

    def genModels(self):
        print("Generating Models...")
        with tf.Session(graph=tf.Graph()) as sess:
            data = tf.placeholder(self.dtype, shape=self.ishape, name=self.iname)
            shape = tf.shape(data)
            shape_sliced = tf.strided_slice(shape, [0,], [1,], strides=[1,], shrink_axis_mask=1)
            shape_stacked = tf.stack([shape_sliced, -1])
            output = tf.reshape(data, shape_stacked, name=self.oname)
            # output = tf.reshape(data, self.oshape, name=self.oname)
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

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)

def test_OP():
    input_shape = (None, 1, 1, 5)  # NHWC
    new_shape = (5,)
    op = OpChecker(input_shape, new_shape)
    op.genModels()
    op.genTFLiteModel()

test_OP()
