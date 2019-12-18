import os
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


def nhwc2nchw(shape):
    return (shape[0], shape[3], shape[1], shape[2])


def oihw2hwoi(shape):
    return (shape[2], shape[3], shape[0], shape[1])


class OpChecker:
    def __init__(self, name, ishape, oshape, dtype="float32"):
        # args in TensorFlow style
        self.name = name
        self.ishape = ishape
        self.oshape = oshape
        self.iname = "data"
        self.dtype = dtype
        self.tflite_model_path = "quant.tflite"

    def genModels(self):
        print("Generating Models...")
        with tf.Session(graph=tf.Graph()) as sess:
            data = tf.placeholder(
                tf.float32,
                shape=self.ishape,
                name=self.iname)
            op = tf.nn.softmax(logits=data, name=self.name)
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
        if self.dtype is "uint8":
            self.input = np.random.uniform(
                low=0, high=255, size=self.ishape).astype(
                self.dtype)
        else:
            self.input = np.random.uniform(size=self.ishape).astype(self.dtype)

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

        interpreter.set_tensor(input_details[0]['index'], self.input)
        interpreter.invoke()
        if (len(self.oshape)) == 4:
            self.output_tflite = interpreter.get_tensor(
                output_details[0]['index']).transpose(0, 3, 1, 2)
        else:
            self.output_tflite = interpreter.get_tensor(
                output_details[0]['index'])

    def runTVM(self, useRemote=False, dumpIR=False):
        if useRemote:
            target = 'llvm -device=arm_cpu -model=mtk6763 -target=aarch64-none-linux-gnueabi -mcpu=cortex-a53 -mattr=+neon'
            target_host = target
            os.environ["TVM_NDK_CC"] = "/home/wzh/alios/prebuilts/gcc/linux-x86/aarch64/aarch64-linux-gnueabi-4.9-glibc-2.20/bin/aarch64-linux-gnueabi-g++"
            rpc_url = "11.163.182.45"
            rpc_port = 20093
            rpc_key = 'mtk6763.quant'
        else:
            target = 'llvm'
            target_host = "llvm"
        import tflite.Model
        print("run TVM...")
        buf = open(self.tflite_model_path, 'rb').read()
        tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
        sym, params = nnvm.frontend.from_tflite(tflite_model)
        target = tvm.target.create(target)
        with nnvm.compiler.build_config(opt_level=3):
            with tvm.build_config(dump_pass_ir=dumpIR):
                graph, lib, params = nnvm.compiler.build(
                    sym, target=target, target_host=target_host, shape={
                        self.iname: self.ishape}, params=params, dtype={
                        self.iname: self.dtype})

        # export deployables
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
        lib.save(os.path.join(bin_path, "lib.ll"), "ll")
        lib.save(os.path.join(bin_path, "lib.asm"), "asm")

        rpc_path = ""
        if useRemote:
            tracker = tvm.rpc.connect_tracker(rpc_url, rpc_port)
            remote = tracker.request(rpc_key, priority=0, session_timeout=30)
            remote.upload(path_so, target=rpc_path + "mylib.so")
            rlib = remote.load_module(rpc_path + "mylib.so")
            mparams = {
                k: tvm.nd.array(
                    v,
                    remote.context(
                        str(target),
                        0)) for k,
                v in params.items()}
            ctx = remote.context(str(target), 0)
            module = runtime.create(graph, rlib, ctx)
        else:
            mparams = params
            ctx = tvm.context(str(target), 0)
            module = runtime.create(graph, lib, ctx)

        module.set_input(self.iname, tvm.nd.array(self.input))
        module.set_input(**mparams)
        module.run()

        self.output_tvm = module.get_output(0, tvm.nd.empty(
            self.oshape, self.dtype, ctx=ctx)).asnumpy()


def test_OP(input_shape, output_shape, dtype, tag="default"):
    print("\n[Begin] Test Op with %s\n" % tag)
    op = OpChecker("softmax", input_shape, output_shape, dtype)
    op.genModels()
    op.preRun()
    # op.runTensorFlow()
    # op.runTVM(True)
    op.runTVM(False)
    op.runTFLite()
    if dtype is "uint8":
        print(op.input)
        print(op.output_tvm)
        print(op.output_tflite)
        np.testing.assert_allclose(op.output_tvm, op.output_tflite)
    else:
        np.testing.assert_allclose(
            op.output_tvm, np.transpose(
                op.output_tf, axes=(
                    0, 3, 1, 2)), atol=1e-3, rtol=1e-3)
    print("\n[Pass] Test OP with %s\n" % tag)


def test():
    print("[START] test")
    print("")
    input_shape = (1, 1001)
    output_shape = input_shape
    dtype = "uint8"
    test_OP(input_shape, output_shape, dtype, tag="softmax")

    print("")
    print("[DONE] test")


test()
