import numpy as np
import onnx
from onnx import helper, numpy_helper
from nnef.binary import read_tensor, write_tensor
from onnx import helper, numpy_helper, ValueInfoProto, NodeProto  # Import ValueInfoProto
from distrubuted_proving.utils import run_inference_on_full_model
# Define core properties function
def tract_core_properties():
    return [("tract_nnef_ser_version", np.array(["0.21.6-pre"], dtype=np.object)),
            ("tract_nnef_format_version", np.array(["beta1"], dtype=np.object))]

# Function to load tensor data from .dat file
def get_tensors_from_dat_file(filepath):
    with open(filepath) as filepath:
        data = read_tensor(filepath)
    return data

# Define the network graph
def build_model():
    # Input
    input_name = 'input_1'
    input_shape = [1, 1, 32, 32]
    
    input_info = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info('_23', onnx.TensorProto.FLOAT, [1])

    # Constants and variables (example paths)
    conv1_weight_path = 'nnef_to_onnx/simple_cnn.nnef/conv1.weight.0.dat'
    conv2_weight_path = 'nnef_to_onnx/simple_cnn.nnef/conv2.weight.0.dat'
    conv2_bias_path = 'nnef_to_onnx/simple_cnn.nnef/conv2.bias.0.dat'
    fc1_weight_path = 'nnef_to_onnx/simple_cnn.nnef/fc1.weight.0.dat'
    fc2_weight_path = 'nnef_to_onnx/simple_cnn.nnef/fc2.weight.0.dat'
    fc3_weight_path = 'nnef_to_onnx/simple_cnn.nnef/fc3.weight.0.dat'
    fc1_gemm_c_add_axis_1_path = 'nnef_to_onnx/simple_cnn.nnef/fc1/Gemm.c_add_axis_1.dat'
    fc2_gemm_c_add_axis_1_path = 'nnef_to_onnx/simple_cnn.nnef/fc2/Gemm.c_add_axis_1.dat'
    fc3_gemm_c_add_axis_1_path = 'nnef_to_onnx/simple_cnn.nnef/fc3/Gemm.c_add_axis_1.dat'

    # Load tensors from .dat files
    conv1_weight = get_tensors_from_dat_file(conv1_weight_path).reshape((6, 1, 3, 3))
    conv1_bias = np.array([0.040680766, -0.06265144, 0.32689142, 0.12525082, 0.1526463, -0.012306213], dtype=np.float32)
    conv2_weight = get_tensors_from_dat_file(conv2_weight_path).reshape((16, 6, 3, 3))
    conv2_bias = get_tensors_from_dat_file(conv2_bias_path)
    fc1_weight = get_tensors_from_dat_file(fc1_weight_path).reshape((120, 576))
    fc2_weight = get_tensors_from_dat_file(fc2_weight_path).reshape((84, 120))
    fc3_weight = get_tensors_from_dat_file(fc3_weight_path).reshape((10, 84))
    fc1_gemm_c_add_axis_1 = get_tensors_from_dat_file(fc1_gemm_c_add_axis_1_path)
    fc2_gemm_c_add_axis_1 = get_tensors_from_dat_file(fc2_gemm_c_add_axis_1_path)
    fc3_gemm_c_add_axis_1 = get_tensors_from_dat_file(fc3_gemm_c_add_axis_1_path)
    
    # Create nodes
    nodes = [
        helper.make_node('Conv', [input_name, 'conv1.weight.0', 'conv1.bias.0'], ['/conv1_Conv_conv'], dilations=[1, 1], strides=[1, 1],  group=1, pads=[0,0]),
        helper.make_node('Relu', ['/conv1_Conv_conv'], ['/Relu_low']),
        helper.make_node('MaxPool', ['/Relu_low'], ['/MaxPool'], kernel_shape=[1, 1, 2, 2], dilations=[1, 1, 1, 1], strides=[1, 1, 2, 2],  pads=[0, 0, 0, 0]),
        helper.make_node('Conv', ['/MaxPool', 'conv2.weight.0', 'conv2.bias.0'], ['/conv2_Conv_conv'], dilations=[1, 1], strides=[1, 1],  group=1, pads=[0, 0, 0, 0]),
        helper.make_node('Relu', ['/conv2_Conv_conv'], ['/Relu_1_low']),
        helper.make_node('MaxPool', ['/Relu_1_low'], ['/MaxPool_1'], kernel_shape=[1, 1, 2, 2], dilations=[1, 1, 1, 1], strides=[1, 1, 2, 2], pads=[0, 0, 0, 0]),
        helper.make_node('Reshape', ['/MaxPool_1'], ['/Reshape_0'], shape=[-1, 576], axis_start=1, axis_count=3),
        helper.make_node('Gemm', ['/Reshape_0', 'fc1.weight.0', '/fc1/Gemm.c_add_axis_1'], ['/fc1_Gemm'], transA=False, transB=True),
        helper.make_node('Relu', ['/fc1_Gemm'], ['/Relu_2_low']),
        helper.make_node('Unsqueeze', ['/Relu_2_low'], ['/fc2_Gemm_ab_add_m'], axes=[0]),
        helper.make_node('Gemm', ['/fc2_Gemm_ab_add_m', 'fc2.weight.0', '/fc2/Gemm.c_add_axis_1'], ['/fc2_Gemm'], transA=False, transB=True),
        helper.make_node('Relu', ['/fc2_Gemm'], ['/Relu_3_low']),
        helper.make_node('Unsqueeze', ['/Relu_3_low'], ['/fc3_Gemm_ab_add_m'], axes=[0]),
        helper.make_node('Gemm', ['/fc3_Gemm_ab_add_m', 'fc3.weight.0', '/fc3/Gemm.c_add_axis_1'], ['_23'], transA=False, transB=True)
    ]
    
    # Create graph
    graph_def = helper.make_graph(nodes, 'network', [input_info], [output_info], initializer=[
        numpy_helper.from_array(conv1_weight, name='conv1.weight.0'),
        numpy_helper.from_array(conv1_bias, name='conv1.bias.0'),
        numpy_helper.from_array(conv2_weight, name='conv2.weight.0'),
        numpy_helper.from_array(conv2_bias, name='conv2.bias.0'),
        numpy_helper.from_array(fc1_weight, name='fc1.weight.0'),
        numpy_helper.from_array(fc2_weight, name='fc2.weight.0'),
        numpy_helper.from_array(fc3_weight, name='fc3.weight.0'),
        numpy_helper.from_array(fc1_gemm_c_add_axis_1, name='/fc1/Gemm.c_add_axis_1'),
        numpy_helper.from_array(fc2_gemm_c_add_axis_1, name='/fc2/Gemm.c_add_axis_1'),
        numpy_helper.from_array(fc3_gemm_c_add_axis_1, name='/fc3/Gemm.c_add_axis_1')
    ])
    
    # Add core properties
    # properties = tract_core_properties()
    # graph_def.doc_string = "version 1.0;"
    # graph_def.properties.extend([helper.make_attribute(key, value) for key, value in properties])
    
    # Create model
    model_def = helper.make_model(graph_def, producer_name='Python-ONNX', opset_imports=[helper.make_opsetid("", 12)])
    
    return model_def

# Build the model
onnx_model = build_model()
# Save the model
model_path ='model.onnx'
onnx.save(onnx_model, model_path)
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
input_path = 'examples/onnx/simple_cnn/input_data.json'

run_inference_on_full_model('model.onnx',input_path)