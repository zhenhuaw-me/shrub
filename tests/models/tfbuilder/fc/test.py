import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

class OpGenerator:
  def __init__(self, M, N, K, tflite_dtype=tf.float32):
    self.ishape   = (M, K)
    self.fshape   = (K, N)
    self.OC = N
    self.tflite_dtype = tflite_dtype
    self.name     = "gen"
    self.iname    = "input"
    self.oname    = "output"

  def genTensorFlowModel(self):
    print("Generating TensorFlow model...")
    with tf.Session(graph=tf.Graph()) as sess:
      net = tf.placeholder(tf.float32, shape=self.ishape, name=self.iname)
      weights = tf.Variable(tf.truncated_normal(self.fshape, stddev=0.1),
                            tf.float32)
      net = tf.matmul(a=net, b=weights)
      bias = tf.Variable(tf.truncated_normal([self.OC,], stddev=1), tf.float32)
      net = tf.nn.bias_add(net, bias, name=self.oname)
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

    M = 1
    N = 6
    K = 8
    tflite_dtype = tf.float32
    # tflite_dtype = tf.uint8

    op = OpGenerator(M, N, K, tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")

genOP()
