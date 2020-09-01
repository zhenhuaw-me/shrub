import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


def _conv_layer(input, dtype, fshape, strides, dilation, padding, oname):
    weights = tf.Variable(tf.truncated_normal(fshape, stddev=0.1), dtype)
    weights = tf.Variable(np.ones(fshape).astype('float32'), dtype)
    net = tf.nn.conv2d(input=input, filter=weights,
                       strides=strides, dilations=dilation,
                       padding=padding, data_format='NHWC',
                       use_cudnn_on_gpu=False,
                       name=oname,
                       )
    # bias = tf.Variable(tf.truncated_normal([fshape[3], ], stddev=1), dtype)
    # net = tf.nn.bias_add(net, bias)
    # net = tf.nn.relu(net, name=oname)
    return net


class OpGenerator:
    def __init__(
            self,
            IHW,
            KHW,
            IC,
            OC,
            stridesHW,
            dilationHW,
            tflite_dtype=tf.float32):
        self.ishape = (1, IHW[0], IHW[1], IC)  # NWHC
        self.fshape = (KHW[0], KHW[1], IC, OC)  # HWIO
        self.OC = OC
        self.strides = (1, stridesHW[0], stridesHW[1], 1)
        self.dilation = (1, dilationHW[0], dilationHW[1], 1)
        self.tflite_dtype = tflite_dtype
        self.name = self._format_naming(tflite_dtype, [IHW[0], KHW[0], IC, OC])
        self.iname = "input"
        self.oname = "output"
        self.padding = "SAME"

    def _format_naming(self, tflite_dtype, attr_list: list):
        dtype_key = 'float32' if tflite_dtype == tf.float32 else 'uint8'
        attrs_key = '.'.join([str(attr) for attr in attr_list])
        return dtype_key + '.' + attrs_key

    def genTensorFlowModel(self):
        print("Generating TensorFlow model...")
        with tf.Session(graph=tf.Graph()) as sess:
            net = tf.placeholder(
                tf.float32,
                shape=self.ishape,
                name=self.iname)
            net = _conv_layer(net, tf.float32, self.fshape, self.strides,
                              self.dilation, self.padding, self.oname)
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
        converter.inference_type = self.tflite_dtype
        converter.inference_inpute_type = self.tflite_dtype
        converter.default_ranges_stats = (-1, 1)
        converter.quantized_input_stats = {self.iname: (0, 1)}

        # converter.post_training_quantize = True
        # converter.target_ops = set([OpsSet.TFLITE_BUILTINS])

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)


def genOP():
    print("[START] genOP\n")
    print("TensorFlow: %s" % tf.__version__)

    IHW = (4, 4)
    KHW = (2, 2)
    IC = 3
    OC = 3
    stridesHW = (1, 1)
    dilationHW = (1, 1)

    tflite_dtype = tf.float32
    tflite_dtype = tf.uint8

    op = OpGenerator(
        IHW,
        KHW,
        IC,
        OC,
        stridesHW,
        dilationHW,
        tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")


genOP()
