# import numpy as np
# from PIL import Image
# 
# def load_labels(filename):
# 
# 
# class Classifier:
#   """ImageNet  classifier"""
#   def __init__(self, model: str, label_file: str):
#       runner_key = model.split('.')[-1]
#       if runner_key == 'tflite':
#         from shrub.tflite import run as runner
#       elif runner_key == 'onnx':
#         from shrub.onnx import run as runner
#       else:
#         raise ValueError("Unsupported runner with model %s" % model)
#       self.runner = runner
#       with open(label_file, 'r') as f:
#           self.labels = [line.strip() for line in f.readlines()]
# 
#   def classify(image):
# 
# 
# 
#   # check the type of the input tensor
#   floating_model = input_details[0]['dtype'] == np.float32
# 
#   # NxHxWxC, H:1, W:2
#   height = input_details[0]['shape'][1]
#   width = input_details[0]['shape'][2]
#   img = Image.open(args.image).resize((width, height))
# 
#   # add N dim
#   input_data = np.expand_dims(img, axis=0)
# 
#   if floating_model:
#     input_data = (np.float32(input_data) - args.input_mean) / args.input_std
# 
#   interpreter.set_tensor(input_details[0]['index'], input_data)
# 
#   interpreter.invoke()
# 
#   output_data = interpreter.get_tensor(output_details[0]['index'])
#   results = np.squeeze(output_data)
# 
#   top_k = results.argsort()[-5:][::-1]
#   labels = load_labels(args.label_file)
#   for i in top_k:
#     if floating_model:
#       print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
#     else:
#       print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
