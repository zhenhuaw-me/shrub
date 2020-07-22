import os
import logging
import shrub

shrub.util.formatLogging(logging.DEBUG)

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(TEST_DIR, '../assets/tests')


def test_run():
    path = os.path.join(ASSETS_DIR, 'mobilenetv2-1.0.onnx')
    m0 = shrub.onnx.parse(path)
    m0.genInput()
    o1 = shrub.onnx.run(path, m0.inputs)
    o2 = shrub.onnx.run(path, m0.inputs)
    assert(shrub.network.cmpTensors(o1, o2))


def test_run_nhwc():
    path = os.path.join(ASSETS_DIR, 'avgpool-nhwc.onnx')
    m0 = shrub.onnx.parse(path, layout='NHWC')
    m0.genInput()
    o1 = shrub.onnx.run(path, m0.inputs, layout='NHWC')
    o2 = shrub.onnx.run(path, m0.inputs, layout='NHWC')
    assert(shrub.network.cmpTensors(o1, o2))


def test_parse():
    path = os.path.join(ASSETS_DIR, 'mobilenetv2-1.0.onnx')
    m = shrub.onnx.parse(path)

    assert(m.name == 'Unknown')
    assert(len(m.inputs) == 1)
    assert(len(m.outputs) == 1)

    assert(m.inputs[0].name == 'data')
    assert(m.inputs[0].dtype == 'float32')
    assert(m.inputs[0].shape == [1, 3, 224, 224])

    assert(m.outputs[0].name == 'mobilenetv20_output_flatten0_reshape0')
    assert(m.outputs[0].dtype == 'float32')
    assert(m.outputs[0].shape == [1, 1000])


if __name__ == '__main__':
    test_parse()
    test_run()
    test_run_nhwc()
