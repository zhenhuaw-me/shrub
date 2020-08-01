import logging

from shrub.common import BaseRunner
from shrub.network import Model, Tensor

logger = logging.getLogger('shrub')


class ONNXRunner(BaseRunner):
    def __init__(self, path: str, layout: str = 'NCHW'):
        """ Runner for ONNX model

        Parameters
        path: path to ONNX model.
        layout: the input output layout of the model.
        """
        super().__init__(path)
        self.layout = layout

    @property
    def quantized(self):
        """ONNX model is not quantized end to end."""
        return False

    def parse(self):
        """Parse a ONNX model to Network semanric."""
        # layout specifies the layout of input/output tensors of the model
        if self.model:
            return self.model
        logger.debug(" parsing %s", self.path)
        import onnxruntime as ort
        TYPE_MAPPING = {
            'tensor(int32)': 'int32',
            'tensor(float)': 'float32',
            'tensor(uint8)': 'uint8',
        }

        sess = ort.InferenceSession(self.path)
        name = 'Unknown'
        dtype = TYPE_MAPPING[sess.get_outputs()[0].type]
        model = Model(name, dtype, layout=self.layout)

        def create_tensor(t):
            return Tensor(t.name, t.shape, TYPE_MAPPING[t.type],
                          layout=self.layout, src_layout=self.layout)

        for t in sess.get_inputs():
            tensor = create_tensor(t)
            model.add('input', tensor)
        for t in sess.get_outputs():
            tensor = create_tensor(t)
            model.add('output', tensor)

        self.model = model
        return model

    def run(self, inputs=None):
        """Run a ONNX model with optional input data.

        Parameters
        inputs: A list of Tensors. If no, generate random inputs.
        """
        logger.debug("running %s", self.path)
        import onnxruntime as ort
        sess = ort.InferenceSession(self.path)
        model = self.parse()
        onames = [t.name for t in model.outputs]

        if inputs is None:
            model.genInput()
            sess.run(onames, model.dict('input', 'data'))
            return None
        else:
            assert (len(inputs) == len(model.inputs))
            input_dict = {}
            for i in range(len(inputs)):
                input_dict[model.inputs[i].name] = inputs[i].dataAs(self.layout)

            outputs = sess.run(onames, input_dict)
            assert (len(outputs) == len(model.outputs))

            for i in range(len(outputs)):
                assert(model.outputs[i].layout == self.layout)
                model.outputs[i].ndarray = outputs[i]
            return model.outputs


def run(path: str, inputs=None, layout='NCHW'):
    """Run a ONNX model with optional input data.

    Parameters
    path: path to ONNX model.
    inputs: A list of Tensors. If no, generate random inputs.
    layout: the input output layout of the model.
    """
    runner = ONNXRunner(path, layout)
    return runner.run(inputs)


def parse(path: str, layout='NCHW'):
    """Parse a ONNX model to Network semanric.

    Parameters
    path: path to ONNX model.
    layout: the input output layout of the model.
    """
    runner = ONNXRunner(path, layout)
    return runner.parse()
