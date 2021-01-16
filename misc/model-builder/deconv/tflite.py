import tensorflow as tf
import numpy as np

IHW = 7
KHW = 3
IC = 2
OC = 4
strides = (2, 2)
dilations = (1, 1)

ISHAPE = (1, IHW, IHW, IC)
OSHAPE = (1, IHW*strides[0], IHW*strides[0], OC)
# OSHAPE = (1, 5, 5, OC)
WSHAPE = (KHW, KHW, OC, IC)

def genWithTFModel():
  class DeconvModule(tf.Module):
    def __init__(self):
      super(DeconvModule, self).__init__()
      initializer = tf.initializers.GlorotUniform()
      self.w = tf.Variable(initializer(shape=WSHAPE), name='weight')
      # w = np.array([1, 2, 3, 4]).astype('float32').reshape(WSHAPE)
      # self.w = tf.Variable(w, name='weight')

    @tf.function(input_signature=[tf.TensorSpec(ISHAPE, tf.float32)])
    def __call__(self, data):
      return tf.nn.conv2d_transpose(data, self.w, OSHAPE, strides, padding='SAME', data_format='NHWC', dilations=None, name='deconv')

  module = DeconvModule()
  tf.saved_model.save(module, 'deconv.saved_model')
  return tf.lite.TFLiteConverter.from_saved_model('deconv.saved_model')

converter = genWithTFModel()
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
