# onnx_splitter.py

import os
import json
from collections import OrderedDict
import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor
# from zkInfer.s3_utils import upload_modelproto_if_not_exists, upload_json_to_s3, upload_modelproto_to_s3
# from zkInfer.utils import compute_bytes_md5_hex
from zkInfer.storage_utils import compute_bytes_md5_hex, upload_to_s3, load_model_proto

def load_json_input(file_path):
    """Load input data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)
    

def format_model_input(input_data_path, session):
    """Format input(s) from JSON to match ONNX model expectations."""
    input_data = load_json_input(input_data_path)["input_data"]
    feed_dict = {}

    for i, inp in enumerate(session.get_inputs()):
        name = inp.name
        dtype = np.float32 if "float" in inp.type else np.int64
        shape = [1 if (dim is None or dim == "batch_size") else dim for dim in inp.shape]

        arr = np.array(input_data[i], dtype=dtype).reshape(shape)
        feed_dict[name] = arr

    return feed_dict


def run_model_inference(onnx_model_path, input_data_path):
    """Run inference with ONNX Runtime using formatted input."""
    session = ort.InferenceSession(onnx_model_path)
    feed_dict = format_model_input(input_data_path, session)
    outputs = session.run(None, feed_dict)
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


def collect_split_inputs_from_inference(onnx_model_path, input_data_path):
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
    feed_dict = format_model_input(input_data_path, session)
    outputs = session.run(None, feed_dict)

     # 5. Collect results
    result_dict = {}
    result_dict.update(feed_dict)  # include inputs
    for out, val in zip(session.get_outputs(), outputs):
        result_dict[out.name] = val

    # result_dict = {input_name: input_data}
    # for out, value in zip(session.get_outputs(), outputs):
    #     result_dict[out.name] = value
    return result_dict


def build_producer_map(model):
    producer_map = {}
    for node in model.graph.node:
        for output in node.output:
            producer_map[output] = node
    return producer_map

def trace_sources(tensor_names, producer_map, passthrough_ops, graph_inputs, initializers):
    sources = set()
    visited = set()

    def dfs(tensor_name):
        if tensor_name in visited:
            return
        visited.add(tensor_name)

        # If it's a graph input → keep
        if tensor_name in graph_inputs:
            sources.add(tensor_name)
            return

        # If it's an initializer (weight) → stop
        if tensor_name in initializers:
            return

        node = producer_map.get(tensor_name, None)
        if node is None:
            sources.add(tensor_name)
            return

        if node.op_type in passthrough_ops:
            # these ops don't "count", trace further back
            for inp in node.input:
                dfs(inp)
        else:
            # real op: stop here, this tensor is a true dependency
            sources.add(tensor_name)

    for t in tensor_names:
        dfs(t)

    return list(sources)


def split_onnx_model(onnx_model_path, split_group_size=1):
    model = onnx.load(onnx_model_path)
    parent_model_hash = compute_bytes_md5_hex(model.SerializeToString())
    e = Extractor(model)

    # boundaries
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {inp.name for inp in model.graph.input}
    producer_map = build_producer_map(model)
    passthrough_ops = {"Identity", "Constant", 'Cast', 'Unsqueeze', 'Slice'}  # don’t split here

    sub_models = []
    counter = 0
    for node in model.graph.node:
        if node.op_type in passthrough_ops:
            continue

        # walk upstream to find real sources
        true_inputs = trace_sources(node.input, producer_map, passthrough_ops, graph_inputs, initializers)
        node_outputs = [o for o in node.output]

        if not node_outputs:
            continue

        sub_model = e.extract_model(true_inputs, node_outputs)
        sub_models.append(sub_model)

    return sub_models, parent_model_hash



def split_onnx_model_with_inputs(onnx_model_path, input_data_path, split_group_size=1):
    split_inputs = collect_split_inputs_from_inference(onnx_model_path, input_data_path)
    sub_models, parent_model_hash = split_onnx_model(onnx_model_path, split_group_size)
    models_with_inputs = []
    for idx, sub_model in enumerate(sub_models):
        flattened_inputs = []
        for inp in sub_model.graph.input:
            if inp.name in split_inputs:
                flattened_inputs.append(split_inputs[inp.name].flatten().tolist())
        if not flattened_inputs:
            continue #no inputs for this sub_model
        sub_mode_input ={'input_data': flattened_inputs}
        md5_hash = f"{parent_model_hash}/{compute_bytes_md5_hex(sub_model.SerializeToString())}"
        sub_model_name = f'sub_model_{idx+1}'
        models_with_inputs.append((sub_model_name, md5_hash, sub_model, sub_mode_input))
    return models_with_inputs

def get_model_info(onnx_model_path):
    model = onnx.load(onnx_model_path)
    model_info = {
            'num_ops': len(model.graph.node),
            'num_params': sum(onnx.numpy_helper.to_array(i).size for i in model.graph.initializer),
            'model_ops': [node.op_type for node in model.graph.node]
        }
    return model_info



if __name__ == "__main__":
    onnx_model_path = "examples/onnx/bert/bert_tiny.onnx"
    input_data_path = "examples/onnx/bert/bert_tiny_input.json"
    split_group_size = 1
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    split_models = split_onnx_model_with_inputs(onnx_model_path, input_data_path, split_group_size)
    for name, md5_hash, model, input_data in split_models:
            savepath = os.path.join(cache_dir, "bert", name)
            model_file_path = os.path.join(savepath, "model.onnx")
            input_file_path = os.path.join(savepath, "input.json")
   
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            with open(model_file_path, "wb") as f:
                f.write(model.SerializeToString())

            os.makedirs(os.path.dirname(input_file_path), exist_ok=True)
            with open(input_file_path, "w") as f:
                json.dump(input_data, f, indent=4)
            print(f"Saved {name}")
    #save global model as well
    parent_model = onnx.load(onnx_model_path)
    parent_model_file_path = os.path.join(cache_dir, "model.onnx")
    with open(parent_model_file_path, "wb") as f:
        f.write(parent_model.SerializeToString())