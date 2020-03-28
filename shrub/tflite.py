import tflite

from . import network
from .common import logger


def run(path: str, inputs=None):
    """Run TFLite, optionally take/return input/output data (Tensor list)."""
    try:
        from tensorflow.lite.python import interpreter as tflite_interp
    except ImportError:
        from tensorflow.contrib.lite.python import interpreter as tflite_interp
    logger.info("running %s", path)

    # prepare runtime
    interp = tflite_interp.Interpreter(model_path=path)
    interp.allocate_tensors()
    idetails, odetails = interp.get_input_details(), interp.get_output_details()
    logger.debug("Inputs: %s", str(idetails))
    logger.debug("Outputs: %s", str(odetails))

    if inputs:
        for i in range(len(inputs)):
            idata = inputs[i].dataAs('NHWC')
            interp.set_tensor(idetails[i]['index'], idata)

        interp.invoke()

        model = parse(path)
        for i in range(len(model.outputs)):
            model.outputs[i].ndarray = interp.get_tensor(odetails[i]['index'])
        return model.outputs
    else:
        interp.invoke()
        return None


def parse(path: str):
    """ Load TFLite model, and build a `Modole` object from it."""
    logger.info("parsing %s", path)
    with open(path, 'rb') as f:
        buf = f.read()
        m = tflite.Model.GetRootAsModel(buf, 0)
    if (m.SubgraphsLength() != 1):
        raise NotImplementedError(
            "Only support one subgraph now, but the model has ",
            m.SubgraphsLength())

    g = m.Subgraphs(0)
    name = 'Unknown' if g.Name() is None else g.Name().decode('utf-8')
    # parse dtype of the model, assume they have same type.
    i0 = g.Tensors(g.Inputs(0)).Type()
    assert (i0 == tflite.TensorType.FLOAT32)
    dtype = 'float32'
    model = network.Model(name, dtype)

    def create_tensor(graph, index):
        t = graph.Tensors(index)
        name = t.Name().decode('utf-8')
        dtype = 'float32'
        shape = t.ShapeAsNumpy()
        return network.Tensor(name, shape, dtype)

    for i in range(g.InputsLength()):
        idx = g.Inputs(i)
        tensor = create_tensor(g, idx)
        model.add('input', tensor)
    for i in range(g.OutputsLength()):
        idx = g.Outputs(i)
        tensor = create_tensor(g, idx)
        model.add('output', tensor)

    return model
