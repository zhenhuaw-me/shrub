import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util


class OpGenerator:
    def __init__(self, tflite_dtype=tf.float32):
        self.tflite_dtype = tflite_dtype
        self.name = "model"
        self.iname = "input"
        self.oname = "output"

    def genTensorFlowModel(self):
        print("Generating TensorFlow model...")
        with tf.Session(graph=tf.Graph()) as sess:
            net = tf.placeholder(self.tflite_dtype, shape=(4, 6, 8, 10), name=self.iname)
            net = tf.strided_slice(net, begin=(0, 0, 0, 0), end=(4, 6, 8, 10), strides=(1, 2, 4, 1),
                                   begin_mask=0, end_mask=0, ellipsis_mask=0,
                                   new_axis_mask=0, shrink_axis_mask=0, name=self.oname)
            sess.run(tf.global_variables_initializer())
            constant_graph = graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [self.oname])
            with tf.gfile.FastGFile(self.name + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def genTFLiteModel(self):
        print("Generating TensorFlow Lite model...")
        converter = tf.lite.TFLiteConverter.from_frozen_graph(
            self.name + ".pb",
            input_arrays=[self.iname],
            output_arrays=[self.oname, ])
        converter.inference_type = self.tflite_dtype
        converter.inference_inpute_type = self.tflite_dtype
        # converter.default_ranges_stats = (0, 6)
        # converter.quantized_input_stats = {self.iname: (100, 100.0)}

        # converter.post_training_quantize = True
        # converter.target_ops = set([OpsSet.TFLITE_BUILTINS])

        tflite_model = converter.convert()
        open(self.name + ".tflite", "wb").write(tflite_model)


def genOP():
    print("[START] genOP\n")
    print("TensorFlow: %s" % tf.__version__)

    tflite_dtype = tf.float32

    op = OpGenerator(tflite_dtype=tflite_dtype)
    op.genTensorFlowModel()
    op.genTFLiteModel()

    print("\n[DONE] genOP")


genOP()
