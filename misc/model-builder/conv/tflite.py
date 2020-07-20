import tensorflow as tf

IHW = (5, 5)
KHW = (2, 2)
IC = 1
OC = 3
strides = (1, 1)
dilations = (1, 1)

data = tf.keras.Input(dtype='float32', name='input', batch_size=1, shape=((IC) + IHW))
kernel = tf.keras.Input(dtype='float32', name='b', batch_size=1, shape=(4, 8, 2))
b1 = tf.keras.layers.Conv2D(OC, strides=strides, padding='same', data_format='channels_last', name='max_pool')(b)

concat = tf.keras.layers.Concatenate(axis=-1, name='output')([a, b1])

model = tf.keras.Model(inputs=[a, b], outputs=[concat])
# print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
