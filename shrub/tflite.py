import numpy as np
import logging
import tflite
from . import nn


def run(model, dumpOutput=False, logLevel=logging.DEBUG):
    """ Run TFLite, optionally load/store data."""
    try:
        from tensorflow.lite.python import interpreter as tflite_interp
    except ImportError:
        from tensorflow.contrib.lite.python import interpreter as tflite_interp
    is_nn_Model = isinstance(model, nn.Model)
    assert is_nn_Model or isinstance(model, str)
    logging.log(logLevel, "Running TFLite...")
    model_path = model.path if is_nn_Model else model
    if not model_path.endswith(model_path):
        logging.warn(
            "%s is not a valid TFLite model path, skip..." %
            model_path)

    interp = tflite_interp.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    idetails, odetails = interp.get_input_details(), interp.get_output_details()
    logging.log(logLevel, "Inputs: %s" % str(idetails))
    logging.log(logLevel, "Outputs: %s" % str(odetails))
    if is_nn_Model:
        for i in range(len(model.inputs)):
            idata = model.inputs[i].dataAs('NHWC')
            interp.set_tensor(idetails[i]['index'], idata)
    interp.invoke()
    if is_nn_Model:
        outs = []
        for i in range(len(model.outputs)):
            out = interp.get_tensor(odetails[i]['index'])
            if dumpOutput:
                nn.store(out, "tflite." + str(i) + ".txt")
            outs.append(out)
        for i in range(len(model.outputs)):
            if model.outputs[i].ndarray is not None:
                msg = "Output %d mismatch!" % i
                if model.dtype == 'uint8':
                    np.testing.assert_allclose(model.outputs[i].dataAs('NHWC'),
                                               outs[i], err_msg=msg)
                else:
                    np.testing.assert_allclose(model.outputs[i].dataAs(
                        'NHWC'), outs[i], atol=1e-3, rtol=1e-3, err_msg=msg)

def parse(path: str):
    """ Load TFLite model, and build a `Modole` object from it."""
    with open(path, 'rb') as f:
        buf = f.read()
        m = tflite.Model.GetRootAsModel(buf, 0)
    if (m.SubgraphsLength() != 1):
        raise NotImplementedError("Only support one subgraph now, but the model has ",
                                  model.SubgraphsLength())

    g = m.Subgraphs(0)
    name = 'Unknown' if g.Name() is None else g.Name().decode('utf-8')
    # parse dtype of the model, assume they have same type.
    i0 = g.Tensors(g.Inputs(0)).Type()
    assert(i0 == tflite.TensorType.FLOAT32)
    dtype = 'float32'
    model = nn.Model(name, dtype)

    def create_tensor(graph, index):
        t = graph.Tensors(index)
        name = t.Name().decode('utf-8')
        dtype = 'float32'
        shape = t.ShapeAsNumpy()
        return nn.Tensor(name, shape, dtype)

    for i in range(g.InputsLength()):
        idx = g.Inputs(i)
        tensor = create_tensor(g, idx)
        model.add('input', tensor)
    for i in range(g.OutputsLength()):
        idx = g.Outputs(i)
        tensor = create_tensor(g, idx)
        model.add('output', tensor)

    return model
