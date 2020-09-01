import os
import numpy as np

import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_frozen_graph(
            'squeezenet.pb',
            input_arrays=['Placeholder',],
            output_arrays=['flatten/Reshape',])
tflite_model = converter.convert()
open("squeezenet.tflite", "wb").write(tflite_model)
