import onnx
import onnx.onnx_cpp2py_export
import numpy as np
import json
import ezkl
import os
import shutil

def read_json_file_to_string(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json.dumps(json_data, indent=4)  # Convert JSON object to a pretty-printed string

def read_json_file_to_dict(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def count_onnx_model_operations(model):
    
    if isinstance(model, str):
        model = onnx.load(model)
    nodes = model.graph.node
    num_operations = len(nodes)
    return num_operations

def count_onnx_model_parameters(model):
    if isinstance(model, str):
        model = onnx.load(model)
    # Initialize the parameter counter
    num_parameters = 0
    # Iterate through all the initializers (weights, biases, etc.)
    for initializer in model.graph.initializer:
        # Get the shape of the parameter
        param_shape = onnx.numpy_helper.to_array(initializer).shape
        # Calculate the total number of elements in this parameter
        param_size = np.prod(param_shape)
        # Add to the total parameter count
        num_parameters += param_size
    return num_parameters

def count_weights_and_tensors_in_onnx_model(model):
    # Load the ONNX model
    if isinstance(model, str):
        model = onnx.load(model)
    
    total_weights = 0
    total_input_size = 0
    total_output_size = 0

    # Count weights in initializers
    for initializer in model.graph.initializer:
        total_weights += len(onnx.numpy_helper.to_array(initializer).flatten())

    # Count input tensor sizes
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        total_input_size += int(np.prod(shape))

    # Count output tensor sizes
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        total_output_size += int(np.prod(shape))

    return total_weights + total_input_size + total_output_size

def get_ezkl_settings(onnx_model, delete_file_afterwards=True):
    """ Generate and return EZKL settings. """
    temp_dir = 'ezkl_settings'
    os.makedirs(temp_dir, exist_ok=True)
    settings_path = os.path.join(temp_dir, 'settings.json')

    if isinstance(onnx_model, str):
        ezkl.gen_settings(onnx_model, settings_path)
    else:
        onnx.save(onnx_model, os.path.join(temp_dir, 'model.onnx'))
        ezkl.gen_settings(os.path.join(temp_dir, 'model.onnx'), settings_path)

    try:
        with open(settings_path, 'r') as f:
            settings_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading JSON settings file: {e}")
        settings_data = {}
    finally:
        if delete_file_afterwards and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
    
    return settings_data

def analyze_onnx_model_for_zk_proving(onnx_model):
    model_ops_count = count_onnx_model_operations(onnx_model)
    model_params_count = count_onnx_model_parameters(onnx_model)
    weights_and_tensor_count = count_weights_and_tensors_in_onnx_model(onnx_model)
    ezkl_settings = get_ezkl_settings(onnx_model, True)
    data_dict = {
        "num_model_ops": model_ops_count,
        "num_model_params": model_params_count,
        "num_model_constants": weights_and_tensor_count,
        "zk_circuit_num_rows": ezkl_settings.get("num_rows", 0),
        "zk_circuit_logrows": ezkl_settings.get("run_args", {}).get("logrows", 0),
        "zk_circuit_total_assignments": ezkl_settings.get("total_assignments", 0),
    }
    return data_dict

def load_onnx_model(model_path):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    return onnx_model

# def get_num_parameters(model = None):
#     # Load the ONNX model
#     if isinstance(model, str):
#         model = onnx.load(model)

#     # Initialize the parameter counter
#     num_parameters = 0
#     # Iterate through all the initializers (weights, biases, etc.)
#     for initializer in model.graph.initializer:
#         # Get the shape of the parameter
#         param_shape = onnx.numpy_helper.to_array(initializer).shape
#         # Calculate the total number of elements in this parameter
#         param_size = np.prod(param_shape)
#         # Add to the total parameter count
#         num_parameters += param_size
    
#     return num_parameters

# def load_onnx_model(model_path):
#     # Load the ONNX model
#     onnx_model = onnx.load(model_path)
#     onnx.checker.check_model(onnx_model)
#     return onnx_model

# def load_and_infer_model(model_path):
#     model = onnx.load(model_path)
#     inferred_model = shape_inference.infer_shapes(model)
#     return inferred_model

# def load_json_input(input_path):
#     with open(input_path, 'r') as f:
#         input_data = json.load(f)
#     # Convert to NumPy array and reshape to (1, 3, 224, 224)
#     if 'net' in input_path:
#         input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is float32
#         if input_data.size != 3 * 224 * 224:
#             raise ValueError(f"Input data must be of size {3 * 224 * 224}, but got {input_data.size}")
#         input_data = input_data.reshape(1, 3, 224, 224)
#         return input_data
#     elif 'mnist_gan' in input_path or 'little_transformer' in input_path:
#         input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is floa
#         return input_data
#     elif 'mnist_classifier' in input_path:
#         input_data = np.array(input_data['input_data'], dtype=np.float32)  # Ensure the data type is floa
#         if input_data.size != 1 * 28 * 28:
#             raise ValueError(f"Input data must be of size {1 * 28 * 28}, but got {input_data.size}")
#         input_data = input_data.reshape(1, 1, 28, 28)

#     if input_data is None:
#         raise ValueError(f"Input data is None")

#     return input_data

# def run_inference_on_split_model(model_parts: List[onnx.ModelProto], input_path: str):

#     inputs_dict = {}
#     # Load and preprocess the JSON input
#     input_data = load_json_input(input_path)

#     model_input_pairs = []

#     # Run inference sequentially on each part
#     for i, model_part in enumerate(model_parts):
#         print(f"Running inference on part {i + 1}...")
#         session = ort.InferenceSession(model_part.SerializeToString())
#         input_names = [input.name for input in session.get_inputs()]

#         if i == 0:
#             inputs_dict[input_names[0]] = input_data

#         assert all(name in inputs_dict for name in input_names), "Input data dictionary keys must match the model input names."

#         infer_input = {}
#         for name in input_names:
#             infer_input[name] = inputs_dict[name]

#         output_names = [output.name for output in session.get_outputs()]

#         results = session.run(output_names, infer_input)
#         if len(results) > 1:
#             pass
#         for name in output_names:
#             inputs_dict[name] = results[0]  # Assuming the output is the first element of results
#         model_input_pairs.append((model_part, infer_input))
#         print(f"Inference results for part {i + 1}:", results)

#     # The final results after running inference on all parts
#     final_results = results[0]
#     print("Final inference results:", final_results)
#     return model_input_pairs, final_results


# def run_inference_on_full_model(model_path, input_path):
#     # Load and check the ONNX model
#     onnx_model:ModelProto = load_onnx_model(model_path)

#     # Load and preprocess the JSON input
#     input_data = load_json_input(input_path)
#     # Run inference
#     session = ort.InferenceSession(onnx_model.SerializeToString())
#     input_name = session.get_inputs()[0].name
#     results = session.run(None, {input_name: input_data})
#     # Print results
#     print("Inference results:", results[0])
#     return results[0]

# def split_onnx_model(onnx_model_path, n_parts=np.inf):
#     def select_cut_points(cutting_points, n_parts):
#         # Ensure n_parts is at least 1
#         n_parts = max(1, n_parts)
#         # Total number of cutting points
#         total_points = len(cutting_points)
#         if n_parts >= total_points:
#             # If n_parts is greater than or equal to total points, return all cutting points
#             return cutting_points
#         # Calculate the indices to pick
#         step = total_points / n_parts
#         selected_indices = [int(step * i) for i in range(1, n_parts)]
#         selected_cut_points = [cutting_points[i] for i in selected_indices]
#         return selected_cut_points

#     if isinstance(onnx_model_path,str):
#         onnx_model = onnx.load(onnx_model_path)
#         onnx_model = shape_inference.infer_shapes(onnx_model)
#     else:
#         onnx_model = onnx_model_path
#         # onnx_model = shape_inference.infer_shapes(onnx_model)
#     spl_onnx = OnnxSplitting(onnx_model, verbose=1, doc_string=False, fLOG=print)
#     cut_points = select_cut_points(spl_onnx.cutting_points, n_parts)

#     parts = split_onnx(onnx_model, cut_points=spl_onnx.cutting_points, verbose=1, stats=False, fLOG=print)

#     return parts

# def split_onnx_mode_oldl(onnx_model_path, num_splits =2):

#     model = onnx.load(onnx_model_path)
#     model = shape_inference.infer_shapes(model)

#     if num_splits <=1:
#         return [model]

#     # Get the total number of layers in the model
#     total_nodes = len(model.graph.node)
#     nodes_per_split = total_nodes // num_splits  # Calculate the number of layers per split

#     # Split the model into parts
#     split_models = []
#     start_index = 0
#     print(f'initial_parameter_count: {get_num_parameters(model=model)}')  # Count parameters before splitting

#     for i in range(num_splits):
#         end_index = min(start_index + nodes_per_split, total_nodes)  # Ensure end index does not exceed total nodes
        
#         part_nodes = model.graph.node[start_index:end_index]
#         next_nodes = model.graph.node[end_index:end_index+nodes_per_split]

#         # Collect initializers that are used as inputs in the next split
#         split_input_names = [inp for node in part_nodes for inp in node.input]
#         part_initializers = [init for init in model.graph.initializer if init.name in split_input_names]

#         # Get the output name of the last node of this part
#         part_output_name = part_nodes[-1].output[0] if part_nodes else None

#         # Create a new ONNX graph for this part
#         part_graph = helper.make_graph(
#             part_nodes,
#             model.graph.name,
#             model.graph.input if i == 0 else [helper.make_tensor_value_info(prev_part_output_name, onnx.TensorProto.FLOAT, None)],
#             [helper.make_tensor_value_info(part_output_name, onnx.TensorProto.FLOAT, None)] if part_output_name else [],
#             part_initializers
#         )

#          # Create ONNX model for this part
#         part_model = helper.make_model(part_graph)
#         split_models.append(part_model)

#         #  # Update for next split
#         start_index = end_index
#         prev_part_output_name = part_output_name
#         print(f'split_{i}_parameter_count: {get_num_parameters(model=part_model)}')
    
#     return split_models

# Function to convert and flatten NumPy arrays to lists
def convert_and_flatten_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.reshape([-1]).tolist()  # Flatten and convert to list
        elif isinstance(value, dict):
            d[key] = convert_and_flatten_ndarray_to_list(value)
    return d

# def main():
#     import os
#     original_model_path = 'examples/onnx/mobilenet/mobilenetv2_050_Opset18.onnx'
#     original_input_path = 'examples/onnx/mobilenet/input.json'
#     split_models = split_onnx_model(original_model_path, n_parts=100)

#     full_model_result =  run_inference_on_full_model(original_model_path,original_input_path)

#     model_input_pairs, final_results = run_inference_on_split_model(split_models, original_input_path)

#     for idx, model_input_pair in enumerate(model_input_pairs):
#             part_model, input = model_input_pair
#             folde_name  = os.path.join('split_model_output', str(idx))
#             os.makedirs(folde_name, exist_ok=True)
#             save_apth = os.path.join(folde_name,f'network_{idx}.onnx')
#             onnx.save_model(part_model,save_apth)
#             # input_flattened = convert_and_flatten_ndarray_to_list(input)

#             with open(os.path.join(folde_name,f"{idx}_input.json"), 'w') as json_file:
#                 json.dump(json.dumps(input), json_file, indent=4)  # indent=4 for pretty-printing
#     pass
#     # run_inference_on_full_model()



if __name__ == "__main__":
    # main()
    pass