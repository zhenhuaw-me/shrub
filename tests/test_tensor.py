#!/usr/bin/env python
import numpy as np
from shrub.network import Tensor


def test_quantize():
    # data
    scale = 0.2
    zero_point = 100
    fp32 = [-1.0, 0, 1.0, 2.0, 30.0]
    uint8 = [95, 100, 105, 110, 250]

    # run
    t = Tensor('test', (len(fp32)), 'float32', quantized=False)
    t.ndarray = np.asarray(fp32).astype('float32')
    t.setQuantizeParam(scale, zero_point)
    t.quantize()

    # verify
    assert(t.scale == scale)
    assert(t.zero_point == zero_point)
    assert(t.dtype == 'uint8')
    quantized = t.ndarray.flatten()
    for i in range(len(uint8)):
        assert(uint8[i] == quantized[i])


def test_dequantize():
    # data
    scale = 0.2
    zero_point = 100
    fp32 = [-1.0, 0, 1.0, 2.0, 30.0]
    uint8 = [95, 100, 105, 110, 250]

    # run
    t = Tensor('test', (len(uint8)), 'uint8', quantized=True)
    t.ndarray = np.asarray(uint8).astype('uint8')
    t.setQuantizeParam(scale, zero_point)
    t.dequantize()

    # verify
    assert(t.scale == scale)
    assert(t.zero_point == zero_point)
    assert(t.dtype == 'float32')
    dequantized = t.ndarray.flatten()
    for i in range(len(fp32)):
        assert(fp32[i] == dequantized[i])


def main():
    test_quantize()
    test_dequantize()


if __name__ == '__main__':
    main()
