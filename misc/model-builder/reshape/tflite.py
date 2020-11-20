import tensorflow as tf
import numpy as np

ISHAPE = (1, 2, 3, 4)
OSHAPE = (int(np.product(ISHAPE)),)


def genWithKeras():
  data = tf.keras.Input(dtype='float32', name='input', batch_size=ISHAPE[0], shape=ISHAPE[1:])
  reshape = tf.keras.layers.Reshape(OSHAPE, name='reshaped')(data)
  model = tf.keras.Model(inputs=[data], outputs=[reshape])
  return tf.lite.TFLiteConverter.from_keras_model(model)


def genWithTFModel():
  class ReshapeModule(tf.Module):
    def __init__(self):
      super(ReshapeModule, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(ISHAPE, tf.float32)])
    def __call__(self, data):
      return tf.reshape(data, OSHAPE)

  module = ReshapeModule()
  tf.saved_model.save(module, 'reshape.saved_model')
  return tf.lite.TFLiteConverter.from_saved_model('reshape.saved_model')

converter = genWithKeras()
# converter = genWithTFModel()
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
