import logging
import numpy as np
from contextlib import contextmanager


def formatJSON(fname: str):
    """Format JSON for readability."""
    import json
    assert fname
    f = open(fname, 'r')
    j = json.load(f)
    f.close()

    f = open(fname, 'w')
    f.write(json.dumps(j, indent=2))
    f.close()


def args(index: int):
    """Get index-th arguments of the program"""
    import sys
    assert index >= 0 and index < len(sys.argv)
    return sys.argv[index]


# from
# https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
@contextmanager
def suppressStdout():
    import sys
    import os
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppressLogging(level="error"):
    logmap = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    new_level = logmap[level]
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(new_level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def formatLogging(level):
    fmt = '%(asctime)s %(levelname).1s [%(name)s][%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=fmt, level=level)


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
