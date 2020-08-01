import tensorflow as tf

IHW = 5
KHW = (2, 2)
IC = 1
OC = 3
strides = (1, 1)
dilations = (1, 1)

data = tf.keras.Input(dtype='float32', name='input', batch_size=1, shape=(IHW, IHW, IC))
conv = tf.keras.layers.Conv2D(OC, KHW, strides=strides, padding='same', data_format='channels_last', name='conv')(data)

model = tf.keras.Model(inputs=[data], outputs=[conv])

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.compat.v1.TFLiteConverter.from_keras_model(model)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
tflite_model = converter.convert()


with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
