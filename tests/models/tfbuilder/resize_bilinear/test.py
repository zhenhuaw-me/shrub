import numpy as np
import jackwish as jw
import os

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util

import tvm
import nnvm
from tvm.contrib import graph_runtime as runtime

class OpChecker:
  def __init__(self, name, ishape, resize_hw_as, align_corners=False, dtype="float32"):
    # args in TensorFlow style
    self.name     = name
    self.ishape   = ishape
    self.resize   = resize_hw_as
    self.oshape   = (ishape[0], resize_hw_as[0], resize_hw_as[1], ishape[3])
    self.align_corners = align_corners
    self.iname    = "input"
    self.oname    = "output"
    self.dtype    = dtype
    self.tflite_model_path = dtype + ".tflite"

  def saveInputs(self, fname="input.txt"):
    print("saving inputs in NCHW %s to %s ..." % (str(jw.nhwc2nchw(self.ishape)), fname))
    size = 1
    for s in self.ishape:
      size = size * s
    to_save = self.input_nchw.reshape((size,))
    np.savetxt(fname, to_save, fmt="%i")

  def saveOutputs(self, fname="output.txt"):
    print("saving outputs in NCHW %s to %s ..." % (str(jw.nhwc2nchw(self.oshape)), fname))
    size = 1
    for s in self.oshape:
      size = size * s
    to_save = self.output_tflite.reshape((size,))
    np.savetxt("output.tflite.txt", to_save, fmt="%i")
    to_save = self.output_tvm.reshape((size,))
    np.savetxt("output.tvm.uint8.txt", np.uint8(to_save), fmt="%i")
    np.savetxt("output.tvm.float.txt", to_save, fmt="%.15f")

  def loadInputs(self, fname="none.txt"):
    print("loading inputs in NCHW %s from %s ..." % (str(jw.nhwc2nchw(self.ishape)), fname))
    loaded = np.loadtxt(fname).astype(self.dtype)
    self.input_nchw = loaded.reshape(jw.nhwc2nchw(self.ishape))
    self.input_nhwc = self.input_nchw.transpose(0, 2, 3, 1)

  def genModels(self):
    print("Generating Models...")
    with tf.Session(graph=tf.Graph()) as sess:
      net = tf.placeholder(tf.float32, shape=self.ishape, name=self.iname)
      net = tf.image.resize_bilinear(images=net, size=self.resize,
                                     align_corners=self.align_corners, name=self.oname)
      sess.run(tf.global_variables_initializer())
      constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [self.oname])
      with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())

  def preRun(self):
    self.input_nhwc = np.random.uniform(low=0, high=255, size=self.ishape).astype(self.dtype)
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

  def runTVM(self, useRemote=False, dumpIR=False):
    if useRemote:
        target = 'llvm -device=arm_cpu -model=mtk6763 -target=aarch64-none-linux-gnueabi -mcpu=cortex-a53 -mattr=+neon'
        target_host = target
        os.environ["TVM_NDK_CC"] = "/home/wzh/alios/prebuilts/gcc/linux-x86/aarch64/aarch64-linux-gnueabi-4.9-glibc-2.20/bin/aarch64-linux-gnueabi-g++"
        rpc_url = "11.163.182.45"
        rpc_port = 20093
        rpc_key = 'mtk6763.alios'
    else:
        target = 'llvm'
        target_host="llvm"
    import tflite.Model
    print("run TVM...")
    buf = open(self.tflite_model_path, 'rb').read()
    tflite_model = tflite.Model.Model.GetRootAsModel(buf, 0)
    sym, params = nnvm.frontend.from_tflite(tflite_model)
    target = tvm.target.create(target)
    with nnvm.compiler.build_config(opt_level=3):
        with tvm.build_config(dump_pass_ir=dumpIR):
            graph, lib, params = nnvm.compiler.build(sym,
                    target=target, target_host=target_host,
                    shape={self.iname : jw.nhwc2nchw(self.ishape)},
                    params=params, dtype={ self.iname : self.dtype })

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
        mparams = {k: tvm.nd.array(v, remote.context(str(target), 0)) for k, v in params.items()}
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
    else:
        mparams = params
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)

    module.set_input(self.iname, tvm.nd.array(self.input_nchw))
    module.set_input(**mparams)
    module.run()

    self.output_tvm = module.get_output(0, tvm.nd.empty(jw.nhwc2nchw(self.oshape), self.dtype, ctx=ctx)).asnumpy()

def testFindTrickyInputs(ishape, resize_hw_as, align_corners, dtype="float32", limits=100, tag="default"):
    print("\n[Begin] Test FindTrickyInputs with %s\n" % tag)
    net = OpChecker("model", ishape, resize_hw_as, align_corners=align_corners, dtype=dtype)
    mismatches = 0

    for i in range(limits):
        print("[Running] Tricky Inputs Searching [%4i/%4i] ..." % (i, limits))
        with jw.suppress_stdout():
            with jw.suppress_logging():
                net.preRun()
                net.runTFLite()
                # net.runTVM(useRemote=True)
                net.runTVM(useRemote=False)
        try:
            np.testing.assert_allclose(net.output_tflite, net.output_tvm)
        except AssertionError:
            print("[Warning] outputs mismatch: %s" % mismatches)
            net.saveInputs(tag + ".inputs." + str(mismatches) + ".txt")
            mismatches = mismatches + 1

    print("\n[END] Test FindTrickyInputs with %s, mismatches %i\n" % (tag, mismatches))

def testMismatchRate(ishape, resize_hw_as, align_corners, dtype="float32", limits=100, tag="default"):
    print("\n[Begin] Test FindTrickyInputs with %s\n" % tag)
    net = OpChecker("model", ishape, resize_hw_as, align_corners=align_corners, dtype=dtype)
    mismatches = 0
    elems = np.prod(net.oshape)

    for i in range(limits):
        print("[Running] Tricky Inputs Searching [%4i/%4i] ..." % (i, limits))
        with jw.suppress_stdout():
            with jw.suppress_logging():
                net.preRun()
                net.runTFLite()
                # net.runTVM(useRemote=True)
                net.runTVM(useRemote=False)
        try:
            np.testing.assert_allclose(net.output_tflite, net.output_tvm)
        except AssertionError:
            out1 = np.uint8(net.output_tvm.reshape((elems,)))
            out2 = np.uint8(net.output_tflite.reshape((elems,)))
            for i in range(elems):
                if (out1[i] != out2[i]):
                    mismatches = mismatches + 1

    print("iters %i" % (limits + 1))
    print("elems %i" % elems)

    print("\n[END] Test FindTrickyInputs with %s, mismatches %i\n" % (tag, mismatches))

def test_OP(input_shape, resize_hw_as, align_corners=True, dtype="float32", input_file=None, tag="default"):
    print("\n[Begin] Test Op with %s\n" % tag)

    op = OpChecker("model", input_shape, resize_hw_as, align_corners=align_corners, dtype=dtype)
    op.genModels()
    return
    if input_file:
        op.loadInputs(input_file)
    else:
        op.preRun()
    # op.runTensorFlow()
    op.runTFLite()
    # op.runTVM(useRemote=True)
    op.runTVM(useRemote=False)
    # np.testing.assert_allclose(op.output_tvm, np.transpose(op.output_tf, axes=(0, 3, 1, 2)),
    #                            atol=1e-3, rtol=1e-3)
    if dtype == "float32":
        np.testing.assert_allclose(op.output_tvm, op.output_tflite, atol=1e-3, rtol=1e-3)
    else:
        try:
            np.testing.assert_allclose(op.output_tflite, np.uint8(op.output_tvm))
            # np.testing.assert_allclose(op.output_tflite, op.output_tvm)
        except AssertionError:
            op.saveInputs()
            op.saveOutputs()
            elems = np.prod(op.oshape)
            tvm_out = op.output_tvm.reshape((elems,))
            tflite_out = op.output_tflite.reshape((elems,))
            for i in range(elems):
                if (tvm_out[i] != tflite_out[i]):
                    print("different result: %s, %s", tvm_out[i], tflite_out[i])
            assert False

    print(op.input_nchw)
    print(op.output_tflite)
    print(op.output_tvm)

    print("\n[Pass] Test OP with %s\n" % tag)

def test():
    print("[START] test")
    print("")

    input_shape  = (1, 32, 64, 40)
    resize_hw_as = (256, 512)
    align_corners=False

    dtype="float32"
    dtype="uint8"
    # os.environ["MY_LLVM_OPT_LEVEL"] = "0"
    # os.environ["TVM_OP_USE_EXTERN"] = "true"

    input_file=None
    # input_file="input.txt"

    # input searching
    limits = 9999
    # testFindTrickyInputs(input_shape, resize_hw_as, align_corners=align_corners, dtype=dtype, limits=limits, tag="resize")
    # testMismatchRate(input_shape, resize_hw_as, align_corners=align_corners, dtype=dtype, limits=limits, tag="resize")

    test_OP(input_shape, resize_hw_as, align_corners=align_corners,
            dtype=dtype, input_file=input_file, tag="resizebilinear")

    print("")
    print("[DONE] test")

test()
