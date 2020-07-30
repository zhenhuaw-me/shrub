import logging
import numpy as np
from PIL import Image

from shrub.tflite import TFLiteRunner
from shrub.onnx import ONNXRunner

logger = logging.getLogger('shrub')


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Classifier:
    """ImageNet classifier"""
    def __init__(self, model: str, label_file: str):
        self.runner_key = model.split('.')[-1]
        if self.runner_key == 'tflite':
            self.runner = TFLiteRunner(model)
            self.mean = 127.5
            self.std = 127.5
        elif self.runner_key == 'onnx':
            self.runner = ONNXRunner(model)
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise ValueError("Unsupported runner with model %s" % model)
        self.quantized = self.runner.quantized
        self.model = self.runner.parse()

        with open(label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def setMeanStd(self, mean, std):
        self.mean = mean
        self.std = std

    def preprocess(self, image):
        spatialShape = self.model.inputs[0].spatialShape()
        img = Image.open(image).resize(spatialShape)
        img = np.expand_dims(img, axis=0)
        if not self.quantized:
            img = img.astype('float32')
            if self.runner_key == 'onnx':
                img = img / 255.0
            img = ((img - self.mean) / self.std).astype('float32')
        return img

    def classify(self, image, top=5):
        logger.debug("classifying %s" % image)
        inputs = self.model.inputs

        inputs[0].setData(self.preprocess(image), layout='NHWC')
        outputs = self.runner.run(inputs)

        output = outputs[0].ndarray
        if self.runner_key == 'onnx':
            output = softmax(output)
        output = output.squeeze()

        topN = output.argsort()[-top:][::-1]
        results = list()
        for i in topN:
            if self.quantized:
                ret = ('{:06.4f}: {}'.format(float(output[i] / 255.0), self.labels[i]))
            else:
                ret = ('{:06.4f}: {}'.format(float(output[i]), self.labels[i]))
            results.append(ret)
        return results
