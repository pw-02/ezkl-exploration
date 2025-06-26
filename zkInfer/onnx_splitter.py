# onnx_splitter.py

import os
import json
from collections import OrderedDict
import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor
from zkInfer.s3_utils import upload_modelproto_if_not_exists, upload_json_to_s3, upload_modelproto_to_s3
from zkInfer.utils import compute_bytes_md5

def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_model_input(input_data_path, expected_shape, input_type, idx=0):
    """Format input tensor from JSON to match ONNX model expectations."""
    expected_shape = [-1 if dim == 'batch_size' else dim for dim in expected_shape]
    input_data = load_json_input(input_data_path)['input_data']
    if 'yolo' in str(input_data_path).lower():
        expected_shape = [1, 3, 416, 416]  # Example for YOLO models
    if input_type == 'tensor(float)':
        reshaped_input = np.array(input_data, dtype=np.float32).reshape(expected_shape)
    elif input_type == 'tensor(int64)':
        reshaped_input = np.array(input_data[idx] if expected_shape else input_data[0][0], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    if 'gpt' in str(input_data_path).lower():
        reshaped_input = np.reshape(input_data, (1, 64))

    return reshaped_input


def run_model_inference(onnx_model_path, input_data_path):
    """Run inference with ONNX Runtime using formatted input."""
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    input_tensor = format_model_input(input_data_path, input_shape, input_type)
    outputs = session.run(None, {input_name: input_tensor})
    return outputs


def extract_model(onnx_model_path, node_inputs, node_outputs):
    if not os.path.exists(onnx_model_path):
        raise ValueError(f"Invalid input model path: {onnx_model_path}")
    if not node_outputs:
        raise ValueError("Output tensor names shall not be empty!")

    model = onnx.load(onnx_model_path)
    e = Extractor(model)
    new_model = e.extract_model(node_inputs, node_outputs)
    return new_model


def merge_onnx_models(sub_models: OrderedDict):
    first_model_id, first_model = next(iter(sub_models.items()))
    merged_model = first_model
    merged_model.graph.ClearField('output')

    sub_model_list = list(sub_models.items())
    for idx, (_, model) in enumerate(sub_model_list[1:]):
        for input_tensor in model.graph.input:
            if all(input_tensor.name != merged_input.name for merged_input in merged_model.graph.input):
                merged_model.graph.input.append(input_tensor)
        model.graph.ClearField('input')

        merged_model.graph.node.extend(model.graph.node)
        merged_model.graph.initializer.extend(model.graph.initializer)

        inputs_seen = set()
        merged_model.graph.input[:] = [i for i in merged_model.graph.input
                                       if i.name not in inputs_seen and not inputs_seen.add(i.name)]

        if idx == len(sub_model_list) - 2:
            for output_tensor in model.graph.output:
                if output_tensor not in merged_model.graph.output:
                    merged_model.graph.output.append(output_tensor)

        for value_info in model.graph.value_info:
            if value_info not in merged_model.graph.value_info:
                merged_model.graph.value_info.append(value_info)

    return merged_model


def collect_intermediate_inference_outputs(onnx_model_path, input_data_path, format_model_input_fn = None):
    model = onnx.load(onnx_model_path)
    model.graph.ClearField('output')
    shape_info = onnx.shape_inference.infer_shapes(model)

    for node in shape_info.graph.node:
        for output_name in node.output:
            if not any(o.name == output_name for o in model.graph.output):
                output_info = onnx.ValueInfoProto()
                output_info.name = output_name
                model.graph.output.append(output_info)

    session = ort.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_type = session.get_inputs()[0].type
    print(f"Input name: {input_name}, shape: {input_shape}, type: {input_type}")
    input_data = format_model_input(input_data_path, input_shape, input_type)

    outputs = session.run(None, {input_name: input_data})
    result_dict = {input_name: input_data}
    for out, value in zip(session.get_outputs(), outputs):
        result_dict[out.name] = value
    return result_dict

def split_onnx_model(onnx_model_path, split_group_size):
    model = onnx.load(onnx_model_path)
    initializers = {init.name for init in model.graph.initializer}
    exclude_operations = {'Identity', 'Constant'}

    all_sub_models = OrderedDict()
    counter = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type in exclude_operations or node.name in initializers:
            continue

        node_inputs = [i for i in node.input if i not in initializers and 'Constant' not in i]
        node_outputs = [o for o in node.output if o not in initializers and 'Constant' not in o]
        
        sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)

        all_sub_models[f'sub_model_{counter+1}'] = sub_model
        counter += 1

    if split_group_size > 1:
        grouped = [dict(list(all_sub_models.items())[i:i + split_group_size])
                   for i in range(0, len(all_sub_models), split_group_size)]
        return [(f"split_group_{i+1}", merge_onnx_models(group)) for i, group in enumerate(grouped)]
    else:
        return list(all_sub_models.items())  # [(name, sub_model), ...]



def prepare_submodel_record(sub_model, intermediate_outputs):
    """Returns (flattened_inputs, raw_bytes, md5_hash, model_metadata) or None if no inputs."""
    flattened_inputs = []
    for inp in sub_model.graph.input:
        if inp.name in intermediate_outputs:
            flattened_inputs.append(intermediate_outputs[inp.name].flatten().tolist())
    if not flattened_inputs:
        return None

    import onnx
    raw_bytes = sub_model.SerializeToString()
    md5_hash = compute_bytes_md5(raw_bytes)
    model_metadata = {
        'name': getattr(sub_model, 'name', 'submodel'),
        'num_ops': len(sub_model.graph.node),
        'num_params': sum(onnx.numpy_helper.to_array(i).size for i in sub_model.graph.initializer),
        'model_ops': [node.op_type for node in sub_model.graph.node]
    }
    return (flattened_inputs, raw_bytes, md5_hash, model_metadata)

def save_split_models_disk(submodels, intermediate_outputs, prefix, overwrite=False):
    from collections import OrderedDict
    import os
    import onnx

    models_with_inputs = OrderedDict()
    for name, sub_model in submodels:
        record = prepare_submodel_record(sub_model, intermediate_outputs)
        if record is None:
            continue
        flattened_inputs, raw_bytes, md5_hash, model_metadata = record

        model_dir = os.path.join(prefix, md5_hash)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.onnx')
        if overwrite or not os.path.exists(model_path):
            onnx.save(sub_model, model_path)
        models_with_inputs[name] = (md5_hash, model_path, flattened_inputs, model_metadata)
    return models_with_inputs

def save_split_models_s3(submodels, intermediate_outputs, s3_bucket, prefix, overwrite=False):
    from collections import OrderedDict

    models_with_inputs = OrderedDict()
    for name, sub_model in submodels:
        record = prepare_submodel_record(sub_model, intermediate_outputs)
        if record is None:
            continue
        flattened_inputs, raw_bytes, md5_hash, model_metadata = record
        model_dir = os.path.join(prefix, md5_hash)
        s3_model_key = os.path.join(model_dir, 'model.onnx')
        if overwrite:
            upload_modelproto_to_s3(raw_bytes, s3_bucket, s3_model_key)
        else:
            upload_modelproto_if_not_exists(raw_bytes, s3_bucket, s3_model_key)
        models_with_inputs[name] = (md5_hash, s3_model_key, flattened_inputs, model_metadata)
    return models_with_inputs
















# def save_split_models_disk(submodels, intermediate_outputs, prefix, overwrite=False):
#     models_with_inputs = OrderedDict()

#     for name, sub_model in submodels:
#         flattened_inputs = []
#         for inp in sub_model.graph.input:
#             if inp.name in intermediate_outputs:
#                 flattened_inputs.append(intermediate_outputs[inp.name].flatten().tolist())

#         if not flattened_inputs:
#             continue

#         md5_hash = compute_bytes_md5(sub_model.SerializeToString())
#         model_dir = os.path.join(prefix, md5_hash)
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(model_dir, 'model.onnx')
#         # input_path = os.path.join(model_dir, 'input.json')
#         onnx.save(sub_model, model_path)
#         # with open(input_path, 'w') as f:
#         #     json.dump({'input_data': flattened_inputs}, f, indent=4)

#         model_metadata = {
#                 'name': name,
#                 'num_ops': len(sub_model.graph.node),
#                 'num_params': sum(onnx.numpy_helper.to_array(i).size for i in sub_model.graph.initializer),
#                 'model_ops': [node.op_type for node in sub_model.graph.node]
#             }

#         models_with_inputs[name] = (md5_hash, model_path, flattened_inputs, model_metadata)

#     return models_with_inputs

# def save_split_models_s3(submodels, intermediate_outputs, s3_bucket, prefix, overwrite=False):
#         models_with_inputs = OrderedDict()

#         for name, sub_model in submodels:
#             flattened_inputs = []
#             for inp in sub_model.graph.input:
#                 if inp.name in intermediate_outputs:
#                     flattened_inputs.append(intermediate_outputs[inp.name].flatten().tolist())

#             if not flattened_inputs:
#                 continue

#             raw_bytes = sub_model.SerializeToString()
#             md5_hash = compute_bytes_md5(raw_bytes)
#             s3_model_key = f"{prefix}/{md5_hash}/model.onnx"

#             if overwrite:
#                 upload_modelproto_to_s3(raw_bytes, s3_bucket, s3_model_key)
#             else:
#                 upload_modelproto_if_not_exists(raw_bytes, s3_bucket, s3_model_key)

#             model_metadata = {
#                 'name': name,
#                 'num_ops': len(sub_model.graph.node),
#                 'num_params': sum(onnx.numpy_helper.to_array(i).size for i in sub_model.graph.initializer),
#                 'model_ops': [node.op_type for node in sub_model.graph.node]
#             }

#             models_with_inputs[name] = (md5_hash, s3_model_key, flattened_inputs, model_metadata)

#         return models_with_inputs
    



def get_model_info(onnx_model_path):
    model = onnx.load(onnx_model_path)
    model_info = {
            'num_ops': len(model.graph.node),
            'num_params': sum(onnx.numpy_helper.to_array(i).size for i in model.graph.initializer),
            'model_ops': [node.op_type for node in model.graph.node]
        }
    return model_info

# def split_model(onnx_model_path, intermediate_outputs, split_group_size, cache_dir):
#     model = onnx.load(onnx_model_path)
#     initializers = {init.name for init in model.graph.initializer}
#     exclude_operations = {'Identity', 'Constant'}

#     all_sub_models = OrderedDict()
#     for idx, node in enumerate(model.graph.node):
#         if node.op_type in exclude_operations or node.name in initializers:
#             continue

#         node_inputs = [i for i in node.input if i not in initializers and 'Constant' not in i]
#         node_outputs = [o for o in node.output if o not in initializers and 'Constant' not in o]
#         sub_model = extract_model(onnx_model_path, node_inputs, node_outputs)
#         all_sub_models[f'split_model_{idx+1}'] = sub_model

#     if split_group_size > 1:
#         grouped = [dict(list(all_sub_models.items())[i:i + split_group_size])
#                    for i in range(0, len(all_sub_models), split_group_size)]
#         all_sub_models = [merge_onnx_models(group) for group in grouped]
#     else:
#         all_sub_models = list(all_sub_models.values())

#     models_with_inputs = OrderedDict()
#     for idx, sub_model in enumerate(all_sub_models):
#         flattened_inputs = []
#         for inp in sub_model.graph.input:
#             flattened_inputs.append(intermediate_outputs[inp.name].flatten().tolist())

#         if not flattened_inputs:
#             continue

#         model_name = f'split_model_{idx+1}'
#         model_dir = os.path.join(cache_dir, model_name)
#         os.makedirs(model_dir, exist_ok=True)

#         model_path = os.path.join(model_dir, 'model.onnx')
#         input_path = os.path.join(model_dir, 'input.json')

#         onnx.save(sub_model, model_path)
#         with open(input_path, 'w') as f:
#             json.dump({'input_data': flattened_inputs}, f, indent=4)

#         models_with_inputs[model_name] = (input_path, model_path)

#     return models_with_inputs
