import logging
from . import nn

def run(model, dumpOutput = False, logLevel = logging.DEBUG):
    """ Run TFLite, optionally load/store data."""
    try:
        from tensorflow.lite.python import interpreter as tflite_interp
    except ImportError:
        from tensorflow.contrib.lite.python import interpreter as tflite_interp
    assert isinstance(model, nn.Model) or isinstance(model, str)
    logging.log(logLevel, "Running TFLite...")
    is_nn_Model = isinstance(model, nn.Model)
    if is_nn_Model:
        interp = tflite_interp.Interpreter(model_path=model.path)
    else:
        interp = tflite_interp.Interpreter(model_path=model)
    interp.allocate_tensors()
    idetails, odetails = interp.get_input_details(), interp.get_output_details()
    logging.log(logLevel, "Inputs: %s" % str(idetails))
    logging.log(logLevel, "Inputs: %s" % str(odetails))
    if is_nn_Model:
        for i in range(len(model.inputs)):
            interp.set_tensor(idetails[i]['index'], model.inputs[i].ndarray)
    interp.invoke()
    if is_nn_Model:
        for i in range(len(model.outputs)):
            model.inputs[i].ndarray = interp.get_tensor(odetails[i]['index'])
            if dumpOutput:
                nn.store(model.outputs[i].ndarray, "tflite."+str(i)+".txt")


