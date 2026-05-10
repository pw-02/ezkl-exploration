import json
import os
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

from zkinfer.storage.io import compute_bytes_md5_hex


PASSTHROUGH_OPS = {"Identity", "Constant", "Cast", "Unsqueeze", "Slice"}


def load_json_input(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_model_input(input_data_path: str, session: ort.InferenceSession) -> Dict[str, np.ndarray]:
    input_data = load_json_input(input_data_path)["input_data"]
    feed_dict = {}

    for idx, model_input in enumerate(session.get_inputs()):
        dtype = np.float32 if "float" in model_input.type else np.int64
        shape = [
            1 if dim is None or dim == "batch_size" else dim
            for dim in model_input.shape
        ]

        feed_dict[model_input.name] = np.array(input_data[idx], dtype=dtype).reshape(shape)

    return feed_dict


def run_model_inference(onnx_model_path: str, input_data_path: str):
    session = ort.InferenceSession(onnx_model_path)
    feed_dict = format_model_input(input_data_path, session)
    return session.run(None, feed_dict)


def extract_model(
    onnx_model_path: str,
    node_inputs: Sequence[str],
    node_outputs: Sequence[str],
):
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Invalid ONNX model path: {onnx_model_path}")

    if not node_outputs:
        raise ValueError("node_outputs must not be empty")

    model = onnx.load(onnx_model_path)
    extractor = Extractor(model)
    return extractor.extract_model(node_inputs, node_outputs)


def merge_onnx_models(sub_models: OrderedDict):
    _, first_model = next(iter(sub_models.items()))
    merged_model = first_model
    merged_model.graph.ClearField("output")

    sub_model_list = list(sub_models.items())

    for idx, (_, model) in enumerate(sub_model_list[1:]):
        for input_tensor in model.graph.input:
            if all(input_tensor.name != existing.name for existing in merged_model.graph.input):
                merged_model.graph.input.append(input_tensor)

        model.graph.ClearField("input")

        merged_model.graph.node.extend(model.graph.node)
        merged_model.graph.initializer.extend(model.graph.initializer)

        inputs_seen = set()
        deduped_inputs = []
        for graph_input in merged_model.graph.input:
            if graph_input.name not in inputs_seen:
                inputs_seen.add(graph_input.name)
                deduped_inputs.append(graph_input)

        merged_model.graph.ClearField("input")
        merged_model.graph.input.extend(deduped_inputs)

        if idx == len(sub_model_list) - 2:
            for output_tensor in model.graph.output:
                if output_tensor not in merged_model.graph.output:
                    merged_model.graph.output.append(output_tensor)

        for value_info in model.graph.value_info:
            if value_info not in merged_model.graph.value_info:
                merged_model.graph.value_info.append(value_info)

    return merged_model


def collect_tensor_values_from_inference(
    onnx_model_path: str,
    input_data_path: str,
) -> Dict[str, np.ndarray]:
    model = onnx.load(onnx_model_path)
    model.graph.ClearField("output")

    shape_info = onnx.shape_inference.infer_shapes(model)

    for node in shape_info.graph.node:
        for output_name in node.output:
            if not any(output.name == output_name for output in model.graph.output):
                output_info = onnx.ValueInfoProto()
                output_info.name = output_name
                model.graph.output.append(output_info)

    session = ort.InferenceSession(model.SerializeToString())
    feed_dict = format_model_input(input_data_path, session)
    outputs = session.run(None, feed_dict)

    tensor_values = dict(feed_dict)

    for output, value in zip(session.get_outputs(), outputs):
        tensor_values[output.name] = value

    return tensor_values


def build_producer_map(model) -> Dict[str, onnx.NodeProto]:
    producer_map = {}

    for node in model.graph.node:
        for output in node.output:
            producer_map[output] = node

    return producer_map


def trace_sources(
    tensor_names: Sequence[str],
    producer_map: Dict[str, onnx.NodeProto],
    passthrough_ops: set,
    graph_inputs: set,
    initializers: set,
) -> List[str]:
    sources = set()
    visited = set()

    def dfs(tensor_name: str) -> None:
        if tensor_name in visited:
            return

        visited.add(tensor_name)

        if tensor_name in graph_inputs:
            sources.add(tensor_name)
            return

        if tensor_name in initializers:
            return

        producer = producer_map.get(tensor_name)
        if producer is None:
            sources.add(tensor_name)
            return

        if producer.op_type in passthrough_ops:
            for input_name in producer.input:
                dfs(input_name)
        else:
            sources.add(tensor_name)

    for tensor_name in tensor_names:
        dfs(tensor_name)

    return list(sources)


def split_onnx_model(
    onnx_model_path: str,
    split_group_size: int = 1,
):
    model = onnx.load(onnx_model_path)
    parent_model_hash = compute_bytes_md5_hex(model.SerializeToString())
    extractor = Extractor(model)

    initializers = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = {graph_input.name for graph_input in model.graph.input}
    producer_map = build_producer_map(model)

    sub_models = []

    for node in model.graph.node:
        if node.op_type in PASSTHROUGH_OPS:
            continue

        true_inputs = trace_sources(
            tensor_names=node.input,
            producer_map=producer_map,
            passthrough_ops=PASSTHROUGH_OPS,
            graph_inputs=graph_inputs,
            initializers=initializers,
        )

        node_outputs = list(node.output)
        if not node_outputs:
            continue

        sub_model = extractor.extract_model(true_inputs, node_outputs)
        sub_models.append(sub_model)

    return sub_models, parent_model_hash


def split_onnx_model_with_inputs(
    onnx_model_path: str,
    input_data_path: str,
    split_group_size: int = 1,
):
    tensor_values = collect_tensor_values_from_inference(
        onnx_model_path=onnx_model_path,
        input_data_path=input_data_path,
    )

    sub_models, parent_model_hash = split_onnx_model(
        onnx_model_path=onnx_model_path,
        split_group_size=split_group_size,
    )

    models_with_inputs = []

    for idx, sub_model in enumerate(sub_models):
        flattened_inputs = []

        for graph_input in sub_model.graph.input:
            if graph_input.name in tensor_values:
                flattened_inputs.append(tensor_values[graph_input.name].flatten().tolist())

        if not flattened_inputs:
            continue

        input_data = {"input_data": flattened_inputs}
        sub_model_hash = compute_bytes_md5_hex(sub_model.SerializeToString())
        cache_key = f"{parent_model_hash}/{sub_model_hash}"
        sub_model_name = f"sub_model_{idx + 1}"

        models_with_inputs.append(
            (sub_model_name, cache_key, sub_model, input_data)
        )

    return models_with_inputs


def get_model_info(onnx_model_path: str) -> Dict:
    model = onnx.load(onnx_model_path)

    return {
        "num_ops": len(model.graph.node),
        "num_params": sum(
            onnx.numpy_helper.to_array(initializer).size
            for initializer in model.graph.initializer
        ),
        "model_ops": [node.op_type for node in model.graph.node],
    }


if __name__ == "__main__":
    onnx_model_path = "examples/onnx/bert/bert_large_squad.onnx"
    input_data_path = "examples/onnx/bert/bert_large_squad_input.json"

    cache_dir = "cache/debug_split"
    os.makedirs(cache_dir, exist_ok=True)

    split_models = split_onnx_model_with_inputs(
        onnx_model_path=onnx_model_path,
        input_data_path=input_data_path,
        split_group_size=1,
    )

    for idx, (name, _, model, input_data) in enumerate(split_models, start=1):
        model_file_path = os.path.join(cache_dir, f"model_{idx}.onnx")
        input_file_path = os.path.join(cache_dir, f"input_{idx}.json")

        with open(model_file_path, "wb") as file:
            file.write(model.SerializeToString())

        with open(input_file_path, "w", encoding="utf-8") as file:
            json.dump(input_data, file, indent=4)

        print(f"Saved {name}")

    parent_model = onnx.load(onnx_model_path)
    parent_model_file_path = os.path.join(cache_dir, "model.onnx")

    with open(parent_model_file_path, "wb") as file:
        file.write(parent_model.SerializeToString())