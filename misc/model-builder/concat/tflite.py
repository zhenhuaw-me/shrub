import tensorflow as tf

a = tf.keras.Input(dtype='float32', name='a', batch_size=1, shape=(2, 3, 4))
b = tf.keras.Input(dtype='float32', name='b', batch_size=1, shape=(2, 3, 5))
c = tf.keras.Input(dtype='float32', name='c', batch_size=1, shape=(2, 3, 6))
# b1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format='channels_last', name='max_pool')(b)

concat = tf.keras.layers.Concatenate(axis=-1, name='output')([a, b, c])

model = tf.keras.Model(inputs=[a, b, c], outputs=[concat])
# print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
