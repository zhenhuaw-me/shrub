import numpy as np

class Tensor:
  def __init__(self, name: str, shape: tuple, dtype: str, layout = 'NHWC',
               src_layout = 'NHWC', ndarray = None):
    self.name = name
    self.dtype = dtype
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
  def _supported_layout(self, layout: str):
    assert layout == 'NCHW' or layout == 'NHWC'
  def _same_layout(self, layout: str):
    return (layout == self.layout)
  def shapeAs(self, layout: str):
    self._supported_layout(layout)
    if self._same_layout(layout):
      return self.shape
    else:
      if layout == 'NCHW':
        return nhwc2nchw(self.shape)
      elif layout == 'NHWC':
        return nchw2nhwc(self.shape)
  def gen(self):
    if self.dtype == 'uint8':
      self.ndarray = np.random.uniform(low=0, high=255, size=self.shape).astype(self.dtype)
    else:
      self.ndarray = np.random.uniform(low=-1, high=1, size=self.shape).astype(self.dtype)
  def dataAs(self, layout: str):
    self._supported_layout(layout)
    if self._same_layout(layout):
      return self.ndarray
    else:
      if layout == 'NCHW':
        return nhwc2nchw(self.ndarray)
      elif layout == 'NHWC':
        return nchw2nhwc(self.ndarray)

class Model:
  def __init__(self, name: str, dtype: str, layout = 'NHWC', path = None):
    self.name = name
    self.dtype = dtype
    self.layout = layout
    self.path = path
    self.clear()

  def getTensors(self, ttype: str):
    if ttype in ['input', 'i']:
      return self.inputs
    elif ttype in ['output', 'o']:
      return self.outputs
    else:
      assert "Unsupported tensor set {}".format(tensor_type)

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
      return { t.name : t.shape for t in tensors }
    elif key == 'dtype':
      return { t.name : t.dtype for t in tensors }
    else:
      assert "Unsupported dict key {}".format(key)

  def genInput(self):
    for i in self.inputs: i.gen()
  def loadInput(self):
    # FIXME
    for i in range(0, len(self.inputs)):
      self.inputs[i].ndarray = load(self.inputs[i].shape,
                                    'input.'+str(i)+'.txt',
                                    self.inputs[i].dtype)
  def store(self, ttype: str):
    tensors = self.getTensors(ttype)
    for i in range(0, len(tensors)):
      tensor = tensors[i]
      if self.layout == 'NCHW':
        ndarray = nchw2nhwc(tensor.ndarray)
      elif self.layout == 'NHWC':
        ndarray = tensor.ndarray
      store(ndarray, ttype+'.'+str(i)+'.txt')

  def clear(self):
    self.inputs = list()
    self.outputs = list()

def nhwc2nchw(shape_or_ndarray):
  if isinstance(shape_or_ndarray, (list, tuple)):
    shape = shape_or_ndarray
    if len(shape) == 4:
      return (shape[0], shape[3], shape[1], shape[2])
    # elif len(shape) == 2:
    #   return (shape[1], shape[0])
    else:
      return shape
  elif isinstance(shape_or_ndarray, np.ndarray):
    nda = shape_or_ndarray
    if len(nda.shape) == 4:
      return nda.transpose(0, 3, 1, 2)
    # elif len(nda.shape) == 2:
    #   return nda.transpose(1, 0)
    else:
      return nda
  else:
    return shape_or_ndarray

def nchw2nhwc(shape_or_ndarray):
  if isinstance(shape_or_ndarray, (list, tuple)):
    shape = shape_or_ndarray
    if len(shape) == 4:
      return (shape[0], shape[2], shape[3], shape[1])
    # elif len(shape) == 2:
    #   return (shape[1], shape[0])
    else:
      return shape
  elif isinstance(shape_or_ndarray, np.ndarray):
    nda = shape_or_ndarray
    if len(nda.shape) == 4:
      return nda.transpose(0, 2, 3, 1)
    # elif len(nda.shape) == 2:
    #   return nda.transpose(1, 0)
    else:
      return nda
  else:
    return shape_or_ndarray

def oihw2hwoi(shape):
  if len(shape) != 4:
    import logging
    logging.warning("oihw2hwoi requires 4D shape")
    return shape
  return (shape[2], shape[3], shape[0], shape[1])

def store(ndarray, fname):
  # caller shall handle layout transform
  ndarray = ndarray.reshape((np.prod(ndarray.shape),))
  if ndarray.dtype in ['float64', 'float32']:
    np.savetxt(fname, ndarray, fmt='%.15f')
  else:
    np.savetxt(fname, ndarray, fmt='%i')

def load(shape, fname, dtype):
  # caller shall handle layout transform
  # image?
  import re
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
        with gfile.FastGFile(pb_path,'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
          sess.graph.as_default()
          tf.import_graph_def(graph_def, name='')
          graph_nodes=[n for n in graph_def.node]

    # select 10- weight tensors with most values
    threshold = 1024
    selected = []
    weights = [n for n in graph_nodes if n.op=='Const']
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
    print("Selected %d weight tensor from %s" % (len(selected), model_path))
    return selected

  def _plotDistribution(weights, figure_path=None):
    for w in weights:
      wv = w.ndarray.reshape(np.prod(w.ndarray.shape))
      sns.distplot(wv, hist = False, kde = True,
                   kde_kws = {'linewidth': 2}, label = w.name)

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
