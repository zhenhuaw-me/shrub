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
                 src_layout=None,
                 ndarray=None):
        self.name = name
        self.dtype = dtype
        self.layout = layout
        self.shape = transform(shape, src_layout, layout)
        self.ndarray = transform(ndarray, src_layout, layout)

        self.quant = QuantParam(quantized=quantized)

    def shapeAs(self, layout: str):
        """Obtain shape with given layout, transform automatically."""
        return transform(self.shape, self.layout, layout)

    def dataAs(self, layout: str):
        """Obtain data with given layout, transform automatically."""
        return transform(self.ndarray, self.layout, layout)

    def setData(self, ndarray, layout):
        self.ndarray = transform(ndarray, layout, self.layout)

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


def cmpTensors(t1, t2, atol=1e-5, rtol=1e-5, useLayout=None):
    """Compare Tensor list data"""
    assert (len(t1) == len(t2))
    for i in range(len(t2)):
        if (useLayout is None):
            assert(t1[i].layout == t2[i].layout)
        dt1 = t1[i].dataAs(useLayout)
        dt2 = t2[i].dataAs(useLayout)
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

    def store(self, ttype: str, useLayout=None):
        """Store inputs/outputs to file named as `{ttype}.{number}.txt'.

        These files are expected to been read by the `loadInput` interface.
        Or for comparation.
        """
        tensors = self.getTensors(ttype)
        for i in range(0, len(tensors)):
            tensor = tensors[i]
            ndarray = tensor.dataAs(useLayout)
            store(ndarray, ttype + '.' + str(i) + '.txt')

    def clear(self):
        self.inputs = list()
        self.outputs = list()


def transform(shape_or_ndarray, srcLayout: str, targetLayout: str):
    if (srcLayout is None or targetLayout is None):
        return shape_or_ndarray
    if (len(srcLayout) != len(targetLayout)):
        return shape_or_ndarray
    if (srcLayout == targetLayout):
        return shape_or_ndarray

    def getPerm(src, target):
        char2index = dict()
        for i in range(len(src)):
            c = src[i]
            char2index[c] = i
        return [char2index[c] for c in target]
    perm = getPerm(srcLayout, targetLayout)

    if isinstance(shape_or_ndarray, (list, tuple)):
        shape = shape_or_ndarray
        return [shape[p] for p in perm]
    elif isinstance(shape_or_ndarray, np.ndarray):
        nda = shape_or_ndarray
        if (len(nda.shape) != len(srcLayout)):
            return nda
        return nda.transpose(perm)
    else:
        assert(False)
        return None


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
        def loadImage(fname):
            import imageio
            return np.asarray(imageio.imread(fname))
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
