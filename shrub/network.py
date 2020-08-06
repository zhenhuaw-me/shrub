"""Utils make it easy to compare data across different frameworks"""
import logging

import numpy as np

logger = logging.getLogger('shrub')


class QuantParam:
    """Quantization Parameter of TensorFlow uint8 approach."""
    def __init__(self, scale=1.0, zero_point=127, quantized=True):
        self.quantized = quantized
        self.set(scale, zero_point)

    def set(self, scale, zero_point):
        if (zero_point < 0 or zero_point > 255):
            raise ValueError("Invalid zero point %s" % zero_point)
        self.scale = scale
        self.zero_point = zero_point

    def asTuple(self):
        return tuple(self.scale, self.zero_point)


class Tensor:
    """The Tensor class to hold semantic and data.

    Class `Tensor` include functionalities like:
    * Semantics for a tensor, name, shape, layout and etc.
    * Data in Numpy.
    """
    def __init__(self,
                 name: str,
                 shape: tuple,
                 dtype: str,
                 quantized: bool = False,
                 layout='NHWC',
                 src_layout='NHWC',
                 ndarray=None):
        self.name = name
        self.dtype = dtype

        # layout attributes
        self.layout = layout
        self._supported_layout(layout)
        self._supported_layout(src_layout)
        if self._same_layout(src_layout):
            self.shape = shape
            self.ndarray = ndarray
        else:
            if self.layout == 'NCHW':
                self.shape = nhwc2nchw(shape)
                self.ndarray = nhwc2nchw(ndarray)
            elif self.layout == 'NHWC':
                self.shape = nchw2nhwc(shape)
                self.ndarray = nchw2nhwc(ndarray)

        self.quant = QuantParam(quantized=quantized)

    def _supported_layout(self, layout: str):
        if layout not in ['NCHW', 'NHWC']:
            raise ValueError("Unsupported layout: %s!" % layout)

    def _same_layout(self, layout: str):
        return (layout == self.layout)

    def _convert_to_layout(self, shape_or_ndarray, layout: str):
        self._supported_layout(layout)
        if self._same_layout(layout):
            return shape_or_ndarray
        else:
            if layout == 'NCHW':
                return nhwc2nchw(shape_or_ndarray)
            elif layout == 'NHWC':
                return nchw2nhwc(shape_or_ndarray)
            else:
                raise ValueError("Shall not reach!")

    def shapeAs(self, layout: str):
        """Obtain shape with given layout, transform automatically."""
        return self._convert_to_layout(self.shape, layout)

    def dataAs(self, layout: str):
        """Obtain data with given layout, transform automatically."""
        return self._convert_to_layout(self.ndarray, layout)

    def setData(self, ndarray, layout):
        self._supported_layout(layout)
        if self._same_layout(layout):
            assert((self.shape == ndarray.shape).all())
            self.ndarray = ndarray
        else:
            if layout == 'NCHW':
                self.ndarray = nchw2nhwc(ndarray)
            elif layout == 'NHWC':
                self.ndarray = nhwc2nchw(ndarray)
            else:
                raise ValueError("Shall not reach!")

    def spatialShape(self):
        if self.layout == 'NCHW':
            return self.shape[2:]
        elif self.layout == 'NHWC':
            return self.shape[1:-1]
        else:
            raise ValueError("Unknwon layout %s" % self.layout)

    @property
    def quantized(self):
        return self.quant.quantized

    def quantize(self):
        """Quantize the tensor data to uint8 if it is not quantized yet

        This will overwrite the `self.ndarray`.
        """
        assert(not self.quantized)
        assert(self.dtype == 'float32')
        assert(self.ndarray is not None)
        fp32 = self.ndarray
        scaled = np.divide(fp32, self.quant.scale)
        shiftted = np.add(scaled, self.quant.zero_point)
        self.ndarray = np.clip(shiftted, 0, 255).astype('uint8')
        self.dtype = 'uint8'

    def dequantize(self):
        """Dequantize the tensor data to float32 if it is quantized.

        This will overwrite the `self.ndarray`.
        """
        assert(self.quantized)
        assert(self.dtype == 'uint8')
        assert(self.ndarray is not None)
        int32 = self.ndarray.astype('int32')
        shiftted = np.subtract(int32, self.quant.zero_point)
        fp32 = np.multiply(shiftted.astype('float32'), self.quant.scale)
        self.ndarray = fp32
        self.dtype = 'float32'

    def gen(self):
        if self.dtype == 'uint8':
            self.ndarray = np.random.uniform(
                low=0, high=255, size=self.shape).astype(self.dtype)
        else:
            self.ndarray = np.random.uniform(
                low=-1, high=1, size=self.shape).astype(self.dtype)


def cmpTensors(t1, t2, atol=1e-5, rtol=1e-5):
    """Compare Tensor list data"""
    assert (len(t1) == len(t2))
    for i in range(len(t2)):
        dt1 = t1[i].dataAs('NHWC')
        dt2 = t2[i].dataAs('NHWC')
        # if np.issubdtype(dt1.dtype, np.integer):
        #     atol = 1
        if not np.allclose(dt1, dt2, atol=atol, rtol=rtol):
            logger.error("Tensor %d mismatch!" % i)
            return False
    return True


class Model:
    """Holding information that needs to run a model.

    Representing basic information for a model, such as inputs and outputs.
    """
    def __init__(self, name: str, dtype: str, layout='NHWC'):
        self.name = name
        self.dtype = dtype
        self.layout = layout
        self.clear()

    def getTensors(self, ttype: str):
        if ttype in ['input', 'i']:
            return self.inputs
        elif ttype in ['output', 'o']:
            return self.outputs
        else:
            assert "Unsupported tensor set {}".format(ttype)

    def add(self, ttype: str, t):
        assert isinstance(t, Tensor) and t.layout == self.layout
        if ttype in ['input', 'i']:
            self.inputs.append(t)
        elif ttype in ['output', 'o']:
            self.outputs.append(t)
        else:
            assert "Unsupported tensor set {}".format(ttype)

    def dict(self, ttype: str, key: str):
        tensors = self.getTensors(ttype)
        if key == 'shape':
            return {t.name: t.shape for t in tensors}
        elif key == 'dtype':
            return {t.name: t.dtype for t in tensors}
        elif key == 'data':
            return {t.name: t.ndarray for t in tensors}
        else:
            assert "Unsupported dict key {}".format(key)

    def genInput(self):
        for i in self.inputs:
            i.gen()

    def loadInput(self):
        """Load inputs from file named as `input.{number}.txt'.

        These files are expected to been written by the `store` interface.
        """
        for i in range(0, len(self.inputs)):
            self.inputs[i].ndarray = load(self.inputs[i].shape,
                                          'input.' + str(i) + '.txt',
                                          self.inputs[i].dtype)

    def store(self, ttype: str):
        """Store inputs/outputs to file named as `{ttype}.{number}.txt'.

        These files are expected to been read by the `loadInput` interface.
        Or for comparation.
        """
        tensors = self.getTensors(ttype)
        for i in range(0, len(tensors)):
            tensor = tensors[i]
            if self.layout == 'NCHW':
                ndarray = nchw2nhwc(tensor.ndarray)
            elif self.layout == 'NHWC':
                ndarray = tensor.ndarray
            store(ndarray, ttype + '.' + str(i) + '.txt')

    def clear(self):
        self.inputs = list()
        self.outputs = list()


def nhwc2nchw(shape_or_ndarray):
    if isinstance(shape_or_ndarray, (list, tuple)):
        shape = shape_or_ndarray
        if len(shape) == 4:
            return (shape[0], shape[3], shape[1], shape[2])
        else:
            return shape
    elif isinstance(shape_or_ndarray, np.ndarray):
        nda = shape_or_ndarray
        if len(nda.shape) == 4:
            return nda.transpose(0, 3, 1, 2)
        else:
            return nda
    else:
        return shape_or_ndarray


def nchw2nhwc(shape_or_ndarray):
    if isinstance(shape_or_ndarray, (list, tuple)):
        shape = shape_or_ndarray
        if len(shape) == 4:
            return (shape[0], shape[2], shape[3], shape[1])
        else:
            return shape
    elif isinstance(shape_or_ndarray, np.ndarray):
        nda = shape_or_ndarray
        if len(nda.shape) == 4:
            return nda.transpose(0, 2, 3, 1)
        else:
            return nda
    else:
        return shape_or_ndarray


def oihw2hwoi(shape):
    if len(shape) != 4:
        logger.warning("oihw2hwoi requires 4D shape")
        return shape
    return (shape[2], shape[3], shape[0], shape[1])


def store(ndarray, fname):
    # caller shall handle layout transform
    ndarray = ndarray.reshape((np.prod(ndarray.shape), ))
    if ndarray.dtype in ['float64', 'float32']:
        np.savetxt(fname, ndarray, fmt='%.15f')
    else:
        np.savetxt(fname, ndarray, fmt='%i')


def load(shape, fname, dtype):
    # caller shall handle layout transform
    # image?
    if fname.endswith('.jpg') or fname.endswith('.png'):
        loaded = loadImage(fname)
        if (len(shape) == 4 and shape[1:] != loaded.shape) or \
           (len(shape) != 4 and shape != loaded.shape):
            assert False, "Unsupported shape %s vs. %s" % \
                          (str(shape), str(loaded.shape))
        assert dtype == 'uint8', "Unsupported dtype %s" % dtype
        return loaded

    # text?
    loaded = np.loadtxt(fname).astype(dtype)
    loaded = loaded.reshape(shape)
    return loaded


def loadImage(fname):
    import imageio
    return np.asarray(imageio.imread(fname))


def plot(model_path, figure_path=None, tensor_count=10):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    from tensorflow.python.framework import tensor_util
    import matplotlib.pyplot as plt
    import seaborn as sns
    from .util import suppressStdout, suppressLogging

    class Weights:
        def __init__(self, name, ndarray):
            self.name = name
            self.ndarray = ndarray

    def _getValuableWeights(pb_path, tensor_count=10):
        # load all weights
        with suppressStdout(), suppressLogging():
            with tf.Session() as sess:
                with gfile.FastGFile(pb_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    graph_nodes = [n for n in graph_def.node]

        # select 10- weight tensors with most values
        threshold = 1024
        selected = []
        weights = [n for n in graph_nodes if n.op == 'Const']
        while True:
            for weight in weights:
                v = tensor_util.MakeNdarray(weight.attr['value'].tensor)
                if (np.prod(v.shape) > threshold):
                    selected.append(Weights(weight.name, v))
            if (len(selected) > tensor_count):
                threshold *= 2
                selected.clear()
            else:
                break
        print("Selected %d weight tensor from %s" %
              (len(selected), model_path))
        return selected

    def _plotDistribution(weights, figure_path=None):
        for w in weights:
            wv = w.ndarray.reshape(np.prod(w.ndarray.shape))
            sns.distplot(wv,
                         hist=False,
                         kde=True,
                         kde_kws={'linewidth': 2},
                         label=w.name)

        plt.legend(prop={'size': 10})
        plt.xlabel("Value Distribution of Selected Tensors")
        plt.ylabel("Density")
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        if figure_path:
            plt.savefig(figure_path)
        else:
            plt.show()

    weights = _getValuableWeights(model_path, tensor_count)
    _plotDistribution(weights, figure_path)
