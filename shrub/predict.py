import logging
import numpy as np
from PIL import Image

from shrub.tflite import TFLiteRunner

logger = logging.getLogger('shrub')


class Classifier:
    """ImageNet classifier"""
    def __init__(self, model: str, label_file: str, std=127.5, mean=127.5):
        self.std = std
        self.mean = mean
        runner_key = model.split('.')[-1]
        if runner_key == 'tflite':
            self.runner = TFLiteRunner(model)
        # elif runner_key == 'onnx':
        #   from shrub.onnx import run as runner
        #   self.runner = runner
        else:
            raise ValueError("Unsupported runner with model %s" % model)
        self.quantized = self.runner.quantized
        self.model = self.runner.parse()

        with open(label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def setStdMean(self, std, mean):
        # MEAN = [0.485, 0.456, 0.406]
        # STD = [0.229, 0.224, 0.225]
        self.std = std
        self.mean = mean

    def preprocess(self, image):
        spatialShape = self.model.inputs[0].spatialShape()
        img = Image.open(image).resize(spatialShape)
        input_data = np.reshape(img, (1, 224, 224, 3))
        if not self.quantized:
            input_data = input_data.astype('float32')
            input_data = ((input_data - self.mean) / self.std).astype('float32')
        return input_data

    def classify(self, image, top=5):
        logger.debug("classifying %s" % image)
        inputs = self.model.inputs
        inputs[0].ndarray = self.preprocess(image)
        outputs = self.runner.run(inputs)

        output = outputs[0].ndarray.flatten()
        topN = output.argsort()[-top:][::-1]
        results = list()
        for i in topN:
            if self.quantized:
                ret = ('{:08.6f}: {}'.format(float(output[i] / 255.0), self.labels[i]))
            else:
                ret = ('{:08.6f}: {}'.format(float(output[i]), self.labels[i]))
            results.append(ret)
        return results
