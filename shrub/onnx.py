import numpy as np
from . import network
from .common import logger


def run(path: str, inputs=None):
    logger.info("[onnx] running %s".format(path))
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    model = parse(path)
    onames = [t.name for t in model.outputs]

    if inputs is None:
        model.genInput()
        sess.run(onames, model.dict('input', 'data'))
        return None
    else:
        assert(len(inputs) == len(model.inputs))
        input_dict = {}
        for i in range(len(inputs)):
            input_dict[model.inputs[i].name] = inputs[i].ndarray

        outputs = sess.run(onames, input_dict)
        assert(len(outputs) == len(model.outputs))

        for i in range(len(outputs)):
            model.outputs[i].ndarray = outputs[i]
        return model.outputs


def parse(path: str):
    logger.info("[onnx] parsing %s".format(path))
    import onnxruntime as ort
    TYPE_MAPPING = {
            'tensor(int32)': 'int32',
            'tensor(float)': 'float32',
            }

    sess = ort.InferenceSession(path)

    name = 'Unknown'
    i0 = sess.get_inputs()[0]
    assert(i0.type == 'tensor(float)')
    dtype = 'float32'
    model = network.Model(name, dtype)

    def create_tensor(t):
        return network.Tensor(t.name, t.shape, TYPE_MAPPING[t.type])

    for t in sess.get_inputs():
        tensor = create_tensor(t)
        model.add('input', tensor)
    for t in sess.get_outputs():
        tensor = create_tensor(t)
        model.add('output', tensor)

    return model
