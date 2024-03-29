import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


class OpGenerator:
    def __init__(self, elementWiseOp, Ashape, Bshape, tflite_dtype=tf.float32):
        self.op = elementWiseOp
        self.Ashape = Ashape
        self.Bshape = Bshape
        self.tflite_dtype = tflite_dtype
        self.name = "gen"
        self.iname = "input"
        self.oname = "output"

    def genTensorFlowModel(self):
        print("Generating TensorFlow model...")
        with tf.Session(graph=tf.Graph()) as sess:
            A = tf.placeholder(self.tflite_dtype, shape=self.Ashape, name='A')
            # B = tf.placeholder(self.tflite_dtype, shape=self.Bshape, name='B')
            B = tf.Variable(tf.truncated_normal(self.Bshape, stddev=0.1), name='B')
            net = self.op(A, B, name=self.oname)
            # net = self.op(A, B, name='bin')
            # net = tf.nn.relu(net, name=self.oname)
            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [self.oname])
            with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def genTFLiteModel(self):
        print("Generating TensorFlow Lite model...")
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            self.name + ".pb",
            # input_arrays=['A', 'B'],
            input_arrays=['A',],
            output_arrays=[self.oname, ])
        converter.inference_type = self.tflite_dtype
        converter.inference_inpute_type = self.tflite_dtype
        # converter.default_ranges_stats = (0, 6)
        # converter.quantized_input_stats = {'A': (100, 100.0),
        #                                    'B': (200, 200.0)}

        # converter.post_training_quantize = True
        # converter.target_ops = set([OpsSet.TFLITE_BUILTINS])

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)


def genOP():
    print("[START] genOP\n")
    print("TensorFlow: %s" % tf.__version__)

    shape = (2, 3, 4, 5)
    Ashape = (3, 4, 5, 6)
    Bshape = (1, 6,)

    elementWiseOp = tf.math.add
    # elementWiseOp = tf.math.subtract
    # elementWiseOp = tf.math.multiply
    # elementWiseOp = tf.math.divide

    tflite_dtype = tf.float32
    # tflite_dtype = tf.uint8

    op = OpGenerator(elementWiseOp, Ashape, Bshape, tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")


genOP()
