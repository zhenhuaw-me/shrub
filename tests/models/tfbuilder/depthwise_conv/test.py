import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.lite.python import lite_constants as constants

class OpGenerator:
  def __init__(self, IHW, KHW, IC, stridesHW, dilationHW, tflite_dtype=constants.FLOAT):
    self.ishape   = (1, IHW[0], IHW[1], IC)  # NWHC
    self.fshape   = (KHW[0], KHW[1], IC, 1)  # HWIM
    self.OC       = IC
    self.strides  = (1, stridesHW[0], stridesHW[1], 1)
    self.dilation = dilationHW
    self.tflite_dtype = tflite_dtype
    self.name     = "gen"
    self.iname    = "input"
    self.oname    = "output"
    self.padding  = "SAME"

  def genTensorFlowModel(self):
    print("Generating TensorFlow model...")
    with tf.Session(graph=tf.Graph()) as sess:
      net = tf.placeholder(tf.float32, shape=self.ishape, name=self.iname)
      weights = tf.Variable(tf.truncated_normal(self.fshape, stddev=0.1),
                            tf.float32)
      net = tf.nn.depthwise_conv2d(input=net, filter=weights,
                                   strides=self.strides, rate=self.dilation,
                                   padding=self.padding, data_format='NHWC')
      bias = tf.Variable(tf.truncated_normal([self.OC,], stddev=1), tf.float32)
      net = tf.nn.bias_add(net, bias)
      net = tf.nn.relu(net, name=self.oname)
      sess.run(tf.global_variables_initializer())
      constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [self.oname])
      with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())

  def genTFLiteModel(self):
    print("Generating TensorFlow Lite model...")
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
                  self.name + ".pb",
                  input_arrays = [self.iname,],
                  output_arrays = [self.oname,])
    converter.inference_type = self.tflite_dtype
    converter.inference_inpute_type = self.tflite_dtype
    converter.default_ranges_stats = (0, 6)
    converter.quantized_input_stats = {self.iname: (100, 123.0)}

    # converter.post_training_quantize = True
    # converter.target_ops = set([OpsSet.TFLITE_BUILTINS])

    tflite_model = converter.convert()
    open(self.name + ".tflite", "wb").write(tflite_model)

def genOP():
    print("[START] genOP\n")
    print("TensorFlow: %s" % tf.__version__)

    IHW = (64, 64)
    KHW = (1, 1)
    IC = 64
    stridesHW = (1, 1)
    dilationHW = (1, 1)
    tflite_dtype = tf.float32
    tflite_dtype = tf.uint8

    op = OpGenerator(IHW, KHW, IC,stridesHW, dilationHW, tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")

genOP()
