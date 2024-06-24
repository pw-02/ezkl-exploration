import onnx
from onnx import helper, numpy_helper
import numpy as np
# Create an empty ONNX model
model = onnx.ModelProto()
model.ir_version = 4

# Create the main graph for the model
graph = model.graph

# Define input tensor
input_name = 'external1'
input_shape = [1, 1, 32, 32]  # Shape of the input tensor
input_tensor = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
graph.input.extend([input_tensor])

# Define initializers (variables)
initializers = {
    'variable1': np.random.randn(6, 1, 3, 3).astype(np.float32),
    'variable2': np.random.randn(1, 6).astype(np.float32),
    'variable3': np.random.randn(16, 6, 3, 3).astype(np.float32),
    'variable4': np.random.randn(1, 16).astype(np.float32),
    'variable5': np.random.randn(120, 576).astype(np.float32),
    'variable6': np.random.randn(1, 120).astype(np.float32),
    'variable7': np.random.randn(84, 120).astype(np.float32),
    'variable8': np.random.randn(1, 84).astype(np.float32),
    'variable9': np.random.randn(10, 84).astype(np.float32),
    'variable10': np.random.randn(1, 10).astype(np.float32)
}

for var_name, var_data in initializers.items():
    tensor = numpy_helper.from_array(var_data, name=var_name)
    graph.initializer.extend([tensor])

# Define operations
conv1 = helper.make_node(
    'Conv', ['external1', 'variable1'], ['conv1'], kernel_shape=[3, 3], strides=[1, 1]
)
relu1 = helper.make_node(
    'Relu', ['conv1'], ['relu1']
)
max_pool1 = helper.make_node(
    'MaxPool', ['relu1'], ['max_pool1'], kernel_shape=[2, 2], strides=[2, 2]
)
conv2 = helper.make_node(
    'Conv', ['max_pool1', 'variable3'], ['conv2'], kernel_shape=[3, 3], strides=[1, 1]
)
relu2 = helper.make_node(
    'Relu', ['conv2'], ['relu2']
)
max_pool2 = helper.make_node(
    'MaxPool', ['relu2'], ['max_pool2'], kernel_shape=[2, 2], strides=[2, 2]
)
reshape1 = helper.make_node(
    'Reshape', ['max_pool2'], ['reshape1'], shape=[-1, 576]
)
linear1 = helper.make_node(
    'Gemm', ['reshape1', 'variable5', 'variable6'], ['linear1'], alpha=1.0, beta=1.0
)
relu3 = helper.make_node(
    'Relu', ['linear1'], ['relu3']
)
linear2 = helper.make_node(
    'Gemm', ['relu3', 'variable7', 'variable8'], ['linear2'], alpha=1.0, beta=1.0
)
relu4 = helper.make_node(
    'Relu', ['linear2'], ['relu4']
)
linear3 = helper.make_node(
    'Gemm', ['relu4', 'variable9', 'variable10'], ['linear3'], alpha=1.0, beta=1.0
)

graph.node.extend([
    conv1, relu1, max_pool1,
    conv2, relu2, max_pool2,
    reshape1, linear1, relu3,
    linear2, relu4, linear3
])

# Define outputs
output_name = 'linear3'
output_tensor = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 10])
graph.output.extend([output_tensor])

# Save the ONNX model to file
onnx.save(model, 'build_onnx_model/example.onnx')

print("ONNX model successfully saved to 'example.onnx'")
