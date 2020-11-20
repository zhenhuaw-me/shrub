import tensorflow as tf
import numpy as np

IHW = 4
KHW = 2
IC = 1
OC = 3
strides = (1, 1)
dilations = (1, 1)

ISHAPE = (1, IHW, IHW, IC)
OSHAPE = (1, IHW*strides[0], IHW*strides[0], OC)
WSHAPE = (KHW, KHW, OC, IC)

def genWithTFModel():
  class DeconvModule(tf.Module):
    def __init__(self):
      super(DeconvModule, self).__init__()
      initializer = tf.initializers.GlorotUniform()
      self.w = tf.Variable(initializer(shape=WSHAPE), name='weight')

    @tf.function(input_signature=[tf.TensorSpec(ISHAPE, tf.float32)])
    def __call__(self, data):
      return tf.nn.conv2d_transpose(data, self.w, OSHAPE, strides, padding='VALID', data_format='NHWC', dilations=None, name='deconv')

  module = DeconvModule()
  tf.saved_model.save(module, 'deconv.saved_model')
  return tf.lite.TFLiteConverter.from_saved_model('deconv.saved_model')

converter = genWithTFModel()
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
