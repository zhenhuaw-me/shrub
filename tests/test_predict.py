import os
import shrub
import logging

shrub.util.formatLogging(logging.DEBUG)

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(TEST_DIR, '../assets/tests')


def test_predict_tflite():
    model = os.path.join(ASSETS_DIR, 'mobilenet_v2_1.0_224.tflite')
    labels = os.path.join(ASSETS_DIR, 'labels.txt')
    image = os.path.join(ASSETS_DIR, 'cat.jpg')
    classifier = shrub.Classifier(model, labels)
    results = classifier.classify(image, 5)
    excepted = [
        '0.460855: 286:Egyptian cat',
        '0.347551: 282:tabby, tabby cat',
        '0.116638: 283:tiger cat',
        '0.001217: 682:notebook, notebook computer',
        '0.001146: 288:lynx, catamount',
    ]
    assert(results == excepted)


if __name__ == '__main__':
    test_predict_tflite()
