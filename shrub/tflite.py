import logging
import tflite

from shrub.common import BaseRunner
from shrub.network import Model, Tensor, QuantParam
from shrub.mapping import DTYPE_TFLITE2NAME

logger = logging.getLogger('shrub')


class TFLiteRunner(BaseRunner):
    def __init__(self, path: str):
        super().__init__(path)

    def _getGraph(self):
        with open(self.path, 'rb') as f:
            buf = f.read()
            m = tflite.Model.GetRootAsModel(buf, 0)
        if (m.SubgraphsLength() != 1):
            raise NotImplementedError(
                "Only support one subgraph now, but the model has ",
                m.SubgraphsLength())
        return m.Subgraphs(0)

    @property
    def quantized(self):
        """If the model is quantized end to end, check first output only."""
        g = self._getGraph()
        o0 = g.Tensors(g.Outputs(0)).Type()
        return (o0 == tflite.TensorType.UINT8)

    def parse(self):
        """ Load TFLite model, and build a `Modole` object from it."""
        if self.model:
            return self.model
        g = self._getGraph()
        name = 'Unknown' if g.Name() is None else g.Name().decode('utf-8')
        dtype = DTYPE_TFLITE2NAME[g.Tensors(g.Outputs(0)).Type()]
        model = Model(name, dtype)

        def create_tensor(graph, index):
            t = graph.Tensors(index)
            name = t.Name().decode('utf-8')
            dtype = DTYPE_TFLITE2NAME[graph.Tensors(graph.Outputs(0)).Type()]
            shape = t.ShapeAsNumpy()
            tensor = Tensor(name, shape, dtype)
            tensor.quant = self._parseTensorQuantParam(index)
            return tensor

        for i in range(g.InputsLength()):
            idx = g.Inputs(i)
            tensor = create_tensor(g, idx)
            model.add('input', tensor)
        for i in range(g.OutputsLength()):
            idx = g.Outputs(i)
            tensor = create_tensor(g, idx)
            model.add('output', tensor)

        self.model = model
        return model

    def _parseTensorQuantParam(self, tensor_index):
        g = self._getGraph()
        t = g.Tensors(tensor_index)
        quant = t.Quantization()
        if (t.Type() == tflite.TensorType.UINT8) and (quant is not None):
            assert(quant.ScaleAsNumpy().size == 1), "Per-tensor support only currently"
            assert(quant.ZeroPointAsNumpy().size == 1), "Per-tensor support only currently"
            scale = float(quant.ScaleAsNumpy()[0])
            zero_point = int(quant.ZeroPointAsNumpy()[0])
            return QuantParam(scale, zero_point)
        else:
            return QuantParam(1.0, 127, quantized=False)

    def parseQuantParam(self, inputs=True):
        """Parse the quantization parameter of inputs/outputs of an model."""
        g = self._getGraph()

        if inputs:
            length = g.InputsLength()
            getIndex = g.Inputs
        else:
            length = g.OutputsLength()
            getIndex = g.Outputs

        params = list()
        for i in range(length):
            idx = getIndex(i)
            param = self._parseTensorQuantParam(idx)
            params.append(param)
        return params

    def run(self, inputs=None):
        """Run TFLite, optionally take/return input/output data (Tensor list)."""
        try:
            from tensorflow.lite.python import interpreter as tflite_interp
        except ImportError:
            from tensorflow.contrib.lite.python import interpreter as tflite_interp
        logger.debug("running %s", self.path)

        # prepare runtime
        interp = tflite_interp.Interpreter(model_path=self.path)
        interp.allocate_tensors()
        idetails, odetails = interp.get_input_details(), interp.get_output_details()
        logger.debug("Inputs: %s", str(idetails))
        logger.debug("Outputs: %s", str(odetails))

        if inputs:
            for i in range(len(inputs)):
                idata = inputs[i].dataAs('NHWC')
                interp.set_tensor(idetails[i]['index'], idata)

            interp.invoke()

            model = self.parse()
            for i in range(len(model.outputs)):
                model.outputs[i].ndarray = interp.get_tensor(odetails[i]['index'])
            return model.outputs
        else:
            interp.invoke()
            return None


def run(path: str, inputs=None):
    """Run TFLite, optionally take/return input/output data (Tensor list)."""
    runner = TFLiteRunner(path)
    return runner.run(inputs)


def parse(path: str):
    """ Load TFLite model, and build a `Modole` object from it."""
    runner = TFLiteRunner(path)
    return runner.parse()


def parseQuantParam(path: str, inputs=True):
    """Parse the quantization parameter of inputs/outputs of an model."""
    runner = TFLiteRunner(path)
    return runner.parseQuantParam(inputs)
