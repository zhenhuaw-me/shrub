import os
import random
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util

import tvm
import nnvm
from tvm.contrib import graph_runtime as runtime
from tvm import rpc
import jackwish as jw


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

    def runTVM(self):
        print("run TVM...")
        tvm_target_name = "llvm"
        mlmodel = coremltools.models.MLModel(self.name + ".mlmodel")
        sym, params = nnvm.frontend.from_coreml(mlmodel)
        target = tvm.target.create(tvm_target_name)
        ctx = tvm.context(tvm_target_name, 0)
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape={self.iname + "__0": nhwc2nchw(self.ishape)}, params=params)

        # export deployables
        useRemote = False
        bin_path = './deploy'
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)
        path_so = os.path.join(bin_path, "lib.so")
        if useRemote:
            from tvm.contrib import ndk
            lib.export_library(path_so, ndk.create_shared)
        else:
            lib.export_library(path_so)
        path_json = os.path.join(bin_path, "graph.json")
        with open(path_json, "w") as fo:
            fo.write(graph.json())
        path_params = os.path.join(bin_path, "param.params")
        with open(path_params, "wb") as fo:
            fo.write(nnvm.compiler.save_param_dict(params))

        module = runtime.create(graph, lib, ctx)
        module.set_input(self.iname + "__0", tvm.nd.array(self.input_nchw))
        module.set_input(**params)
        module.run()

        self.output_tvm = module.get_output(0, tvm.nd.empty(
            nhwc2nchw(self.oshape), self.dtype)).asnumpy()

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

    # op.quantizeCoreML()
    # if input_file:
    #     op.loadInputs(input_file)
    # else:
    #     op.preRun()
    # op.cmpQuantizedCoreML()
    # op.runTensorFlow()
    # op.runTVM()
    # op.runTFLite()
    # np.testing.assert_allclose(op.output_tvm, np.transpose(op.output_tf, axes=(0, 3, 1, 2)),
    #                            atol=1e-3, rtol=1e-3)
    # op.saveOutputs()

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
