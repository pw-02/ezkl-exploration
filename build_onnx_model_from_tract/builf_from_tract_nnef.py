import onnx
import numpy as np
from onnx import helper, numpy_helper

# Create an empty ONNX model
model = onnx.ModelProto()
model.ir_version = 4

# Create the main graph for the model
graph = model.graph

# Define input tensor
input_name = 'input_1'
input_shape = [1, 1, 32, 32]  # Shape of the input tensor
input_tensor = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
graph.input.extend([input_tensor])

# Define initializers (variables)
initializers = {
    'conv1.weight.0': np.random.randn(6, 1, 3, 3).astype(np.float32),
    'conv1.bias.0': np.array([0.040680766, -0.06265144, 0.32689142, 0.12525082, 0.1526463, -0.012306213], dtype=np.float32),
    'conv2.weight.0': np.random.randn(16, 6, 3, 3).astype(np.float32),
    'conv2.bias.0': np.random.randn(16).astype(np.float32),
    'fc1.weight.0': np.random.randn(120, 576).astype(np.float32),
    '/fc1/Gemm.c_add_axis_1': np.random.randn(1, 120).astype(np.float32),
    'fc2.weight.0': np.random.randn(84, 120).astype(np.float32),
    '/fc2/Gemm.c_add_axis_1': np.random.randn(1, 84).astype(np.float32),
    'fc3.weight.0': np.random.randn(10, 84).astype(np.float32),
    '/fc3/Gemm.c_add_axis_1': np.random.randn(1, 10).astype(np.float32)
}

for var_name, var_data in initializers.items():
    tensor = numpy_helper.from_array(var_data, name=var_name)
    graph.initializer.extend([tensor])

# Define operations
conv1 = helper.make_node(
    'Conv', ['input_1', 'conv1.weight.0', 'conv1.bias.0'], ['/conv1_Conv_conv'], kernel_shape=[3, 3], strides=[1, 1]
)
relu1 = helper.make_node(
    'Relu', ['/conv1_Conv_conv'], ['/Relu_low']
)
max_pool1 = helper.make_node(
    'MaxPool', ['/Relu_low'], ['/MaxPool'], kernel_shape=[2, 2], strides=[2, 2]
)
conv2 = helper.make_node(
    'Conv', ['/MaxPool', 'conv2.weight.0', 'conv2.bias.0'], ['/conv2_Conv_conv'], kernel_shape=[3, 3], strides=[1, 1]
)
relu2 = helper.make_node(
    'Relu', ['/conv2_Conv_conv'], ['/Relu_1_low']
)
max_pool2 = helper.make_node(
    'MaxPool', ['/Relu_1_low'], ['/MaxPool_1'], kernel_shape=[2, 2], strides=[2, 2]
)
reshape1 = helper.make_node(
    'Reshape', ['/MaxPool_1'], ['/Reshape_0'], shape=[-1, 576], axis_start=1, axis_count=3
)
fc1 = helper.make_node(
    'Gemm', ['/Reshape_0', 'fc1.weight.0', '/fc1/Gemm.c_add_axis_1'], ['/fc1_Gemm']
)
relu3 = helper.make_node(
    'Relu', ['/fc1_Gemm'], ['/Relu_2_low']
)
fc2 = helper.make_node(
    'Gemm', ['/Relu_2_low', 'fc2.weight.0', '/fc2/Gemm.c_add_axis_1'], ['/fc2_Gemm']
)
relu4 = helper.make_node(
    'Relu', ['/fc2_Gemm'], ['/Relu_3_low']
)
fc3 = helper.make_node(
    'Gemm', ['/Relu_3_low', 'fc3.weight.0', '/fc3/Gemm.c_add_axis_1'], ['_23']
)

graph.node.extend([
    conv1, relu1, max_pool1,
    conv2, relu2, max_pool2,
    reshape1, fc1, relu3,
    fc2, relu4, fc3
])

# Define outputs
output_name = '_23'
output_tensor = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 10])
graph.output.extend([output_tensor])

# Save the ONNX model to file
onnx.save(model, 'build_onnx_model/example.onnx')

print("ONNX model successfully saved to 'example.onnx'")
