# NHWC -> transpose -> NCHW -> Pooling -> NCHW -> transpose -> NHWC

import onnx
from onnx import helper, TensorProto

N = 1
C = 2
H = 4
W = 4

shape_nchw = (N, C, H, W)
shape_nhwc = (N, H, W, C)

inhwc = helper.make_tensor_value_info('input_nhwc', TensorProto.FLOAT, shape_nhwc)
inchw = helper.make_tensor_value_info('input_nchw', TensorProto.FLOAT, shape_nchw)
onchw = helper.make_tensor_value_info('output_nchw', TensorProto.FLOAT, shape_nchw)
onhwc = helper.make_tensor_value_info('output_nhwc', TensorProto.FLOAT, shape_nhwc)

input_transpose = helper.make_node('Transpose', ['input_nhwc'], ['input_nchw'], perm=[0, 3, 1, 2])
pooling = helper.make_node('AveragePool', ['input_nchw'], ['output_nchw'], kernel_shape=[1, 1], strides=[1, 1], auto_pad='SAME_UPPER')
output_transpose = helper.make_node('Transpose', ['output_nchw'], ['output_nhwc'], perm=[0, 2, 3, 1])

graph = helper.make_graph([input_transpose, pooling, output_transpose], 'test', [inhwc], [onhwc], value_info=[inhwc, inchw, onchw, onhwc])

model = helper.make_model(graph)

onnx.save(model, '1.onnx')
