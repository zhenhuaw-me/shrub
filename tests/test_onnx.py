import shrub
import logging

shrub.util.formatLogging(logging.DEBUG)

path = shrub.testing.download('mobilenetv2-1.0.onnx')


def test_run():
    m0 = shrub.onnx.parse(path)
    m0.genInput()
    o1 = shrub.onnx.run(path, m0.inputs)
    o2 = shrub.onnx.run(path, m0.inputs)
    assert(shrub.network.cmpTensors(o1, o2))


def test_parse():
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
