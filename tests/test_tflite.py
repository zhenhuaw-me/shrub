import os
import logging
import shrub

shrub.util.formatLogging(logging.DEBUG)

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(TEST_DIR, '../assets/tests')


def test_parse():
    path = os.path.join(ASSETS_DIR, 'mobilenet_v2_1.0_224.tflite')
    m = shrub.tflite.parse(path)

    assert(m.name == 'Unknown')
    assert(len(m.inputs) == 1)
    assert(len(m.outputs) == 1)

    assert(m.inputs[0].name == 'input')
    assert(m.inputs[0].dtype == 'float32')
    assert((m.inputs[0].shape == [1, 224, 224, 3]).all())

    assert(m.outputs[0].name == 'MobilenetV2/Predictions/Reshape_1')
    assert(m.outputs[0].dtype == 'float32')
    assert((m.outputs[0].shape == [1, 1001]).all())


def test_run():
    path = os.path.join(ASSETS_DIR, 'mobilenet_v2_1.0_224.tflite')
    m0 = shrub.tflite.parse(path)
    m0.genInput()
    o1 = shrub.tflite.run(path, m0.inputs)
    o2 = shrub.tflite.run(path, m0.inputs)
    assert(shrub.network.cmpTensors(o1, o2))


if __name__ == '__main__':
    test_parse()
    test_run()
