import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


def _conv_layer(input, dtype, fshape, strides, dilation, padding, oname):
    weights = tf.Variable(np.ones(fshape).astype('float32'), dtype)


class OpGenerator:
    def __init__(self, ishape, oshape, tflite_dtype=tf.float32):
        self.ishape = ishape
        self.oshape = oshape
        self.tflite_dtype = tflite_dtype
        self.name = "model"
        self.iname = "input"
        self.oname = "output"

    def genTensorFlowModel(self):
        print("Generating TensorFlow model...")
        with tf.Session(graph=tf.Graph()) as sess:
            net = tf.placeholder(tf.float32, shape=self.ishape, name=self.iname)
            net = tf.math.abs(net, 'abs')
            net = tf.reshape(net, self.oshape, self.oname)
            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [self.oname])
            with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def genTFLiteModel(self):
        print("Generating TensorFlow Lite model...")
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            self.name + ".pb",
            input_arrays=[self.iname, ],
            output_arrays=[self.oname, ])

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)


def genOP():
    print("[START] genOP\n")
    print("TensorFlow: %s" % tf.__version__)

    ISHAPE = (1, 2, 3, 4)
    OSHAPE = (int(np.product(ISHAPE)),)
    tflite_dtype = tf.float32

    op = OpGenerator(ISHAPE, OSHAPE, tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")


genOP()
