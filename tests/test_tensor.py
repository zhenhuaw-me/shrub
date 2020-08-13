import numpy as np
from shrub.network import Tensor, QuantParam, transform


def test_quantize():
    # data
    scale = 0.2
    zero_point = 100
    fp32 = [-1.0, 0, 1.0, 2.0, 30.0]
    uint8 = [95, 100, 105, 110, 250]

    # run
    t = Tensor('test', (len(fp32)), 'float32', quantized=False)
    t.ndarray = np.asarray(fp32).astype('float32')
    t.quant = QuantParam(scale, zero_point, False)
    t.quantize()

    # verify
    assert(t.quant.scale == scale)
    assert(t.quant.zero_point == zero_point)
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
    t.quant = QuantParam(scale, zero_point, True)
    t.dequantize()

    # verify
    assert(t.quant.scale == scale)
    assert(t.quant.zero_point == zero_point)
    assert(t.dtype == 'float32')
    dequantized = t.ndarray.flatten()
    for i in range(len(fp32)):
        assert(fp32[i] == dequantized[i])

def test_transform():
    assert(transform([1, 2, 3, 4], 'NCHW', 'NHWC') == [1, 3, 4, 2])
    assert(transform([1, 2, 3, 4], 'NHWC', 'NCHW') == [1, 4, 2, 3])
    assert(transform([1, 2, 3], 'ABC', 'CAB') == [3, 1, 2])

    inda = np.arange(120).reshape(2, 3, 4, 5)
    onda = transform(inda, 'NHWC', 'NCHW')
    assert(np.equal(onda, inda.transpose(0, 3, 1, 2)).all())

    onda = transform(inda, 'NCHW', 'NHWC')
    assert(np.equal(onda, inda.transpose(0, 2, 3, 1)).all())


if __name__ == '__main__':
    test_transform()
    test_quantize()
    test_dequantize()
