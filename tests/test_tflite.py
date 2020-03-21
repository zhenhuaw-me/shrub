import shrub

def test_parse():
    path = '../3rdparty/mobilenet_v1_1.0_224.tflite'
    m = shrub.tflite.parse(path)

    assert(m.name == 'Unknown')
    assert(len(m.inputs) == 1)
    assert(len(m.outputs) == 1)

    assert(m.inputs[0].name == 'input')
    assert(m.inputs[0].dtype == 'float32')
    assert((m.inputs[0].shape == [1, 224, 224, 3]).all())

    print(m.outputs[0].name )
    print(m.outputs[0].dtype )
    print(m.outputs[0].shape )


    assert(m.outputs[0].name == 'MobilenetV1/Predictions/Reshape_1')
    assert(m.outputs[0].dtype == 'float32')
    assert((m.outputs[0].shape == [1, 1001]).all())

test_parse()
