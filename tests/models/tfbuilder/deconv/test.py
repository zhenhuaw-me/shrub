import numpy as np
import tfcoreml
import coremltools

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util

import tvm
import nnvm
from tvm.contrib import graph_runtime as runtime
from tvm import rpc


def nhwc2nchw(shape):
    return (shape[0], shape[3], shape[1], shape[2])


def oihw2hwoi(shape):
    return (shape[2], shape[3], shape[0], shape[1])


class ConvOpChecker:
    def __init__(
        self, name, ishape, oshape, fshape, strides=(
            1, 1, 1, 1), dilation=(
            1, 1, 1, 1), dtype="float32"):
        # args in TensorFlow style
        self.name = name
        self.ishape = ishape
        self.oshape = oshape
        self.fshape = fshape
        self.dtype = dtype
        self.iname = "data"
        self.fname = "filter"
        self.strides = strides
        # self.padding  = padding
        self.dtype = dtype
        self.dilation = dilation

    def genModels(self):
        print("Generating Models...")
        with tf.Session(graph=tf.Graph()) as sess:
            data = tf.placeholder(
                tf.float32,
                shape=self.ishape,
                name=self.iname)
            Weights = tf.Variable(
                tf.truncated_normal(
                    self.fshape,
                    stddev=0.1),
                name=self.fname)

            deconv = tf.nn.conv2d_backprop_input(
                input_sizes=self.oshape,
                filter=Weights,
                out_backprop=data,
                strides=list(
                    self.strides),
                padding="SAME",
                use_cudnn_on_gpu=False,
                data_format='NHWC',
                dilations=list(
                    self.dilation),
                name=self.name)

            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [self.name])

            with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

        coreml_model = tfcoreml.convert(
            tf_model_path=self.name + ".pb",
            mlmodel_path=self.name + ".mlmodel",
            input_name_shape_dict={
                self.iname + ":0": self.ishape},
            output_feature_names=[
                self.name + ":0"])

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
                    "import/" + self.name + ":0")]
            self.output_tf = sess.run(
                outputs, feed_dict={
                    image_input_tensor: self.input_nhwc})[0]

    # def runTVM(self, rpc=False, host="0.0.0.0", port="9090"):
    def runTVM(self):
        print("run TVM...")
        tvm_target_name = "llvm -device=arm_cpu -target=aarch64-linux-gnu"
        tvm_target_name = "cuda"
        tvm_target_name = "llvm"
        mlmodel = coremltools.models.MLModel(self.name + ".mlmodel")
        sym, params = nnvm.frontend.from_coreml(mlmodel)
        target = tvm.target.create(tvm_target_name)
        ctx = tvm.context(tvm_target_name, 0)
        with nnvm.compiler.build_config(opt_level=1):
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape={self.iname + "__0": nhwc2nchw(self.ishape)}, params=params)

        module = runtime.create(graph, lib, ctx)
        module.set_input(self.iname + "__0", tvm.nd.array(self.input_nchw))
        module.set_input(**params)
        module.run()

        self.output_tvm = module.get_output(0, tvm.nd.empty(
            nhwc2nchw(self.oshape), self.dtype)).asnumpy()


# NHWC style shape
filter_shape = (3, 3, 256, 256)
padding = "SAME"
if True:
    # yet wrong result
    strides = (1, 4, 4, 1)
    input_shape = (1, 25, 25, 256)
    output_shape = (1, 97, 100, 256)
    # output_shape = (1, 98, 100, 256)
    # output_shape = (1, 99, 100, 256)
    # output_shape = (1, 30, 42, 256)
    # output_shape = (1, 29, 44, 256)
else:
    # same result
    strides = (1, 2, 2, 1)
    input_shape = (1, 30, 46, 256)
    output_shape = (1, 60, 92, 256)

op = ConvOpChecker(
    "deconv",
    input_shape,
    output_shape,
    filter_shape,
    strides=strides)
op.genModels()
op.preRun()
op.runTensorFlow()
op.runTVM()

# np.testing.assert_allclose(op.input_nchw, np.transpose(op.input_nhwc, axes=(0, 3, 1, 2)), atol=1e-3, rtol=1e-3)

np.testing.assert_allclose(
    op.output_tvm,
    np.transpose(
        op.output_tf,
        axes=(
            0,
            3,
            1,
            2)),
    atol=1e-3,
    rtol=1e-3)
