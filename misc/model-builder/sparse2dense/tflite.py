import os
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util


class OpChecker:
    def __init__(self, dense_shape, dtype="float32"):
        # args in TensorFlow style
        self.name = "model"
        self.oshape = dense_shape
        self.iname = "input"
        self.oname = "output"
        self.dtype = dtype
        self.tflite_model_path = dtype + ".tflite"
        self.model_dir = "./model"

    def genModels(self):
        print("Generating Models...")
        dense_shape = self.oshape

        class MyOP(tf.Module):
            @tf.function(
                input_signature=[
                    tf.TensorSpec(
                        shape=(
                            3, 3), dtype=tf.int64, name='indices'), tf.TensorSpec(
                        shape=(
                            3,), dtype=tf.float32, name='values')])
            def myop(self, indices, values):
                spi = tf.sparse.SparseTensor(indices, values, dense_shape)
                return tf.sparse.to_dense(spi, name="output")
                # return indices + values
        net = MyOP()
        tf.saved_model.save(net, self.model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_dir)
        tflite_model = converter.convert()
        open(self.dtype + ".tflite", "wb").write(tflite_model)

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

    def runTFLite(self):
        try:
            from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
        except ImportError:
            from tensorflow.lite.python import interpreter as interpreter_wrapper
        print("run TensorFlow Lite...")
        interpreter = interpreter_wrapper.Interpreter(
            model_path=(self.tflite_model_path))
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
            self.output_tflite = interpreter.get_tensor(
                output_details[0]['index']).transpose(0, 3, 1, 2)
        else:
            self.output_tflite = interpreter.get_tensor(
                output_details[0]['index'])


def test_OP(dense_shape, dtype="float32"):
    print("\n[Begin] Test OP\n")

    op = OpChecker(dense_shape, dtype=dtype)
    op.genModels()

    print("\n[Pass] Test OP\n")


def test_wrapper():
    print("[START] test_wrapper")
    print("")

    dense_shape = (8, 8)
    # dtype = "uint8"
    dtype = "float32"
    test_OP(dense_shape, dtype=dtype)

    print("")
    print("[DONE] test_wrapper")


test_wrapper()
