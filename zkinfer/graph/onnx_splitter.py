import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor

from zkinfer.storage.io import compute_bytes_md5_hex


PASSTHROUGH_OPS = {"Identity", "Constant", "Cast", "Unsqueeze", "Slice"}


@dataclass
class ModelPartition:
    name: str
    input_names: List[str]
    output_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_json_input(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_model_input(
    input_data_path: str,
    session: ort.InferenceSession,
) -> Dict[str, np.ndarray]:
    input_data = load_json_input(input_data_path)["input_data"]
    feed_dict: Dict[str, np.ndarray] = {}

    for idx, model_input in enumerate(session.get_inputs()):
        dtype = np.float32 if "float" in model_input.type else np.int64
        shape = [
            1 if dim is None or dim == "batch_size" else dim
            for dim in model_input.shape
        ]

        feed_dict[model_input.name] = np.array(
            input_data[idx],
            dtype=dtype,
        ).reshape(shape)

    return feed_dict


def run_model_inference(
    onnx_model_path: str,
    input_data_path: str,
):
    session = ort.InferenceSession(onnx_model_path)
    feed_dict = format_model_input(input_data_path, session)
    return session.run(None, feed_dict)


def simplify_onnx_model(
    onnx_model_path: str,
    output_path: str,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> Tuple[str, bool]:
    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise ImportError(
            "onnxsim is required for simplify_model=True. "
            "Install with `pip install onnxsim`."
        ) from exc

    model = onnx.load(onnx_model_path)

    if input_shapes:
        simplified_model, check = simplify(
            model,
            overwrite_input_shapes=input_shapes,
        )
    else:
        simplified_model, check = simplify(model)

    if not check:
        raise RuntimeError(f"ONNX simplification failed validation for {onnx_model_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(simplified_model, output_path)

    return output_path, check


def maybe_simplify_onnx_model(
    onnx_model_path: str,
    simplify_model: bool = False,
    simplified_model_path: Optional[str] = None,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> str:
    if not simplify_model:
        return onnx_model_path

    if simplified_model_path is None:
        root, ext = os.path.splitext(onnx_model_path)
        simplified_model_path = f"{root}_simplified{ext}"

    simplified_path, _ = simplify_onnx_model(
        onnx_model_path=onnx_model_path,
        output_path=simplified_model_path,
        input_shapes=input_shapes,
    )

    return simplified_path


def extract_model(
    onnx_model_path: str,
    node_inputs: Sequence[str],
    node_outputs: Sequence[str],
):
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Invalid ONNX model path: {onnx_model_path}")

    if not node_inputs:
        raise ValueError("node_inputs must not be empty")

    if not node_outputs:
        raise ValueError("node_outputs must not be empty")

    model = onnx.load(onnx_model_path)
    extractor = Extractor(model)
    return extractor.extract_model(list(node_inputs), list(node_outputs))


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


def build_producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
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

    return sorted(sources)


def get_partition_io(
    nodes: Sequence[onnx.NodeProto],
    producer_map: Dict[str, onnx.NodeProto],
    graph_inputs: set,
    initializers: set,
    passthrough_ops: set,
) -> Tuple[List[str], List[str]]:
    node_outputs = set()
    raw_inputs = []

    for node in nodes:
        node_outputs.update(node.output)
        raw_inputs.extend(node.input)

    true_inputs = trace_sources(
        tensor_names=raw_inputs,
        producer_map=producer_map,
        passthrough_ops=passthrough_ops,
        graph_inputs=graph_inputs,
        initializers=initializers,
    )

    outputs = []
    for node in nodes:
        for output_name in node.output:
            if output_name:
                outputs.append(output_name)

    return sorted(set(true_inputs)), outputs


def split_by_single_ops(
    model: onnx.ModelProto,
    passthrough_ops: Optional[set] = None,
) -> List[ModelPartition]:
    passthrough_ops = passthrough_ops or PASSTHROUGH_OPS

    initializers = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = {graph_input.name for graph_input in model.graph.input}
    producer_map = build_producer_map(model)

    partitions: List[ModelPartition] = []

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type in passthrough_ops:
            continue

        input_names, output_names = get_partition_io(
            nodes=[node],
            producer_map=producer_map,
            graph_inputs=graph_inputs,
            initializers=initializers,
            passthrough_ops=passthrough_ops,
        )

        if not input_names or not output_names:
            continue

        partitions.append(
            ModelPartition(
                name=f"sub_model_{len(partitions) + 1}",
                input_names=input_names,
                output_names=output_names,
                metadata={
                    "split_strategy": "single_ops",
                    "node_index": node_idx,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "num_nodes": 1,
                },
            )
        )

    return partitions


def split_by_fixed_groups(
    model: onnx.ModelProto,
    group_size: int,
    passthrough_ops: Optional[set] = None,
) -> List[ModelPartition]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    passthrough_ops = passthrough_ops or PASSTHROUGH_OPS

    initializers = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = {graph_input.name for graph_input in model.graph.input}
    producer_map = build_producer_map(model)

    meaningful_nodes = [
        node for node in model.graph.node
        if node.op_type not in passthrough_ops
    ]

    partitions: List[ModelPartition] = []

    for group_start in range(0, len(meaningful_nodes), group_size):
        group_nodes = meaningful_nodes[group_start: group_start + group_size]

        input_names, output_names = get_partition_io(
            nodes=group_nodes,
            producer_map=producer_map,
            graph_inputs=graph_inputs,
            initializers=initializers,
            passthrough_ops=passthrough_ops,
        )

        if not input_names or not output_names:
            continue

        partitions.append(
            ModelPartition(
                name=f"sub_model_{len(partitions) + 1}",
                input_names=input_names,
                output_names=output_names,
                metadata={
                    "split_strategy": "fixed_groups",
                    "group_size": group_size,
                    "group_start": group_start,
                    "num_nodes": len(group_nodes),
                    "op_types": [node.op_type for node in group_nodes],
                    "node_names": [node.name for node in group_nodes],
                },
            )
        )

    return partitions


def split_onnx_model(
    onnx_model_path: str,
    split_mode: str = "single_ops",
    split_group_size: int = 1,
    simplify_model: bool = False,
    simplified_model_path: Optional[str] = None,
    simplify_input_shapes: Optional[Dict[str, List[int]]] = None,
):
    model_path_for_splitting = maybe_simplify_onnx_model(
        onnx_model_path=onnx_model_path,
        simplify_model=simplify_model,
        simplified_model_path=simplified_model_path,
        input_shapes=simplify_input_shapes,
    )

    model = onnx.load(model_path_for_splitting)
    parent_model_hash = compute_bytes_md5_hex(model.SerializeToString())
    extractor = Extractor(model)

    if split_mode in ("single", "single_op", "single_ops"):
        partitions = split_by_single_ops(model)

    elif split_mode in ("fixed", "fixed_groups"):
        partitions = split_by_fixed_groups(
            model=model,
            group_size=split_group_size,
        )

    else:
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}. "
            "Expected one of: single_ops, fixed."
        )

    sub_models = []

    for partition in partitions:
        sub_model = extractor.extract_model(
            partition.input_names,
            partition.output_names,
        )
        sub_models.append((partition, sub_model))

    metadata = {
        "source_model_path": onnx_model_path,
        "model_path_for_splitting": model_path_for_splitting,
        "simplify_model": simplify_model,
        "split_mode": split_mode,
        "split_group_size": split_group_size,
        "num_partitions": len(sub_models),
    }

    return sub_models, parent_model_hash, metadata


def materialize_submodels_with_inputs(
    sub_models: Sequence[Tuple[ModelPartition, onnx.ModelProto]],
    tensor_values: Dict[str, np.ndarray],
    parent_model_hash: str,
):
    models_with_inputs = []

    for idx, (partition, sub_model) in enumerate(sub_models):
        flattened_inputs = []

        for graph_input in sub_model.graph.input:
            if graph_input.name in tensor_values:
                flattened_inputs.append(
                    tensor_values[graph_input.name].flatten().tolist()
                )

        if not flattened_inputs:
            continue

        input_data = {"input_data": flattened_inputs}
        sub_model_hash = compute_bytes_md5_hex(sub_model.SerializeToString())
        cache_key = f"{parent_model_hash}/{sub_model_hash}"

        models_with_inputs.append(
            (
                partition.name or f"sub_model_{idx + 1}",
                cache_key,
                sub_model,
                input_data,
            )
        )

    return models_with_inputs


def split_onnx_model_with_inputs(
    onnx_model_path: str,
    input_data_path: str,
    split_group_size: int = 1,
    split_mode: str = "single_ops",
    simplify_model: bool = False,
    simplified_model_path: Optional[str] = None,
    simplify_input_shapes: Optional[Dict[str, List[int]]] = None,
):
    model_path_for_splitting = maybe_simplify_onnx_model(
        onnx_model_path=onnx_model_path,
        simplify_model=simplify_model,
        simplified_model_path=simplified_model_path,
        input_shapes=simplify_input_shapes,
    )

    tensor_values = collect_tensor_values_from_inference(
        onnx_model_path=model_path_for_splitting,
        input_data_path=input_data_path,
    )

    sub_models, parent_model_hash, _ = split_onnx_model(
        onnx_model_path=model_path_for_splitting,
        split_mode=split_mode,
        split_group_size=split_group_size,
        simplify_model=False,
    )

    return materialize_submodels_with_inputs(
        sub_models=sub_models,
        tensor_values=tensor_values,
        parent_model_hash=parent_model_hash,
    )


def get_model_info(onnx_model_path: str) -> Dict[str, Any]:
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
    onnx_model_path = "examples/onnx/mnist_classifier/network.onnx"
    input_data_path = "examples/onnx/mnist_classifier/input.json"

    cache_dir = "cache/debug_split"
    os.makedirs(cache_dir, exist_ok=True)

    split_models = split_onnx_model_with_inputs(
        onnx_model_path=onnx_model_path,
        input_data_path=input_data_path,
        split_mode="fixed",
        split_group_size=1,
        simplify_model=True,
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