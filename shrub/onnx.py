from . import network
from .common import logger


def run(path: str, inputs=None, layout='NCHW'):
    """Run a ONNX model with optional input data.

    Parameters
    path: path to ONNX model.
    inputs: A list of Tensors. If no, generate random inputs.
    layout: the input output layout of the model.
    """
    logger.info("running %s", path)
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    model = parse(path, layout=layout)
    onames = [t.name for t in model.outputs]

    if inputs is None:
        model.genInput()
        sess.run(onames, model.dict('input', 'data'))
        return None
    else:
        assert (len(inputs) == len(model.inputs))
        input_dict = {}
        for i in range(len(inputs)):
            input_dict[model.inputs[i].name] = inputs[i].dataAs(layout)

        outputs = sess.run(onames, input_dict)
        assert (len(outputs) == len(model.outputs))

        for i in range(len(outputs)):
            assert(model.outputs[i].layout == layout)
            model.outputs[i].ndarray = outputs[i]
        return model.outputs


def parse(path: str, layout='NCHW'):
    """Parse a ONNX model to Network semanric.

    Parameters
    path: path to ONNX model.
    layout: the input output layout of the model.
    """
    # layout specifies the layout of input/output tensors of the model
    logger.info(" parsing %s", path)
    import onnxruntime as ort
    TYPE_MAPPING = {
        'tensor(int32)': 'int32',
        'tensor(float)': 'float32',
    }

    sess = ort.InferenceSession(path)

    name = 'Unknown'
    i0 = sess.get_inputs()[0]
    assert (i0.type == 'tensor(float)')
    dtype = 'float32'
    model = network.Model(name, dtype, layout=layout)

    def create_tensor(t):
        return network.Tensor(t.name, t.shape, TYPE_MAPPING[t.type],
                              layout=layout, src_layout=layout)

    for t in sess.get_inputs():
        tensor = create_tensor(t)
        model.add('input', tensor)
    for t in sess.get_outputs():
        tensor = create_tensor(t)
        model.add('output', tensor)

    return model
