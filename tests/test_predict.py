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
        "0.4609: 286:Egyptian cat",
        "0.3476: 282:tabby, tabby cat",
        "0.1166: 283:tiger cat",
        "0.0012: 682:notebook, notebook computer",
        "0.0011: 288:lynx, catamount",
    ]
    assert(results == excepted)


def test_predict_onnx():
    model = os.path.join(ASSETS_DIR, 'mobilenetv2-1.0.onnx')
    labels = os.path.join(ASSETS_DIR, 'synset.txt')
    image = os.path.join(ASSETS_DIR, 'cat.jpg')
    classifier = shrub.Classifier(model, labels)
    results = classifier.classify(image, 5)
    excepted = [
        "0.5937: n02123045 tabby, tabby cat",
        "0.3653: n02124075 Egyptian cat",
        "0.0380: n02123159 tiger cat",
        "0.0006: n02971356 carton",
        "0.0006: n03642806 laptop, laptop computer",
    ]
    assert(results == excepted)


if __name__ == '__main__':
    test_predict_tflite()
    test_predict_onnx()
