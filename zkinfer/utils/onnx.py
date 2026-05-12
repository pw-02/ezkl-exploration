import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnx
import onnxruntime as ort

from zkinfer.storage.io import load_json

logger = logging.getLogger(__name__)


def to_numpy_dtype(onnx_runtime_type: str) -> np.dtype:
    dtype_map = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }

    if onnx_runtime_type not in dtype_map:
        raise ValueError(f"Unsupported ONNX Runtime type: {onnx_runtime_type}")

    return dtype_map[onnx_runtime_type]


def resolve_shape(
    shape: Sequence[Any],
    batch_size: int = 1,
    default_dim: int = 1,
    seq_len: int = 128,
) -> List[int]:
    resolved = []

    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif idx == 0:
            resolved.append(batch_size)
        elif dim in {"sequence_length", "seq_len"}:
            resolved.append(seq_len)
        else:
            resolved.append(default_dim)

    return resolved


def _generate_model_inputs(
    session: ort.InferenceSession,
    batch_size: int = 1,
    default_dim: int = 1,
    seq_len: int = 128,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs = {}

    for model_input in session.get_inputs():
        dtype = to_numpy_dtype(model_input.type)
        shape = resolve_shape(
            model_input.shape,
            batch_size=batch_size,
            default_dim=default_dim,
            seq_len=seq_len,
        )

        if model_input.name in {"attention_mask", "input_ids"}:
            value = np.ones(shape, dtype=np.int64)
        elif model_input.name == "token_type_ids":
            value = np.zeros(shape, dtype=np.int64)
        elif dtype == np.bool_:
            value = np.ones(shape, dtype=dtype)
        elif np.issubdtype(dtype, np.integer):
            value = np.ones(shape, dtype=dtype)
        else:
            value = rng.standard_normal(shape).astype(dtype)

        inputs[model_input.name] = value

    return inputs


def load_or_generate_inputs(
    session: ort.InferenceSession,
    input_data_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    if input_data_path is None:
        return _generate_model_inputs(session)

    payload = load_json(input_data_path)

    if "input_data" not in payload:
        raise ValueError(f"Input JSON must contain an 'input_data' field: {input_data_path}")

    input_data = payload["input_data"]
    model_inputs = session.get_inputs()

    if len(input_data) != len(model_inputs):
        raise ValueError(
            f"Input JSON has {len(input_data)} inputs, but model expects {len(model_inputs)}."
        )

    inputs = {}

    for idx, model_input in enumerate(model_inputs):
        dtype = to_numpy_dtype(model_input.type)
        shape = resolve_shape(model_input.shape)

        inputs[model_input.name] = np.asarray(
            input_data[idx],
            dtype=dtype,
        ).reshape(shape)

    return inputs


def inputs_to_json(
    inputs: Dict[str, np.ndarray],
    input_names: Sequence[str],
) -> Dict[str, Any]:
    return {
        "input_data": [
            inputs[name].flatten().tolist()
            for name in input_names
        ]
    }


def simplify_model_if_requested(
    model_path: str,
    simplify: bool = False,
    output_path: Optional[str] = None,
    input_shapes: Optional[Dict[str, List[int]]] = None,
) -> str:
    if not simplify:
        return model_path

    try:
        from onnxsim import simplify as simplify_onnx
    except ImportError as exc:
        raise ImportError(
            "onnxsim is required for simplify=True. "
            "Install it with `pip install onnxsim`."
        ) from exc

    if output_path is None:
        root, ext = os.path.splitext(model_path)
        output_path = f"{root}_simplified{ext}"

    model = onnx.load(model_path)

    simplified_model, ok = (
        simplify_onnx(model, overwrite_input_shapes=input_shapes)
        if input_shapes
        else simplify_onnx(model)
    )

    if not ok:
        raise RuntimeError(f"ONNX simplification failed for {model_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(simplified_model, output_path)

    return output_path


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        logger.warning("ONNX shape inference failed: %s", exc)
        return model


def get_model_info(model_path: str) -> Dict[str, Any]:
    model = onnx.load(model_path)

    return {
        "num_ops": len(model.graph.node),
        "num_params": sum(
            onnx.numpy_helper.to_array(initializer).size
            for initializer in model.graph.initializer
        ),
        "model_ops": [node.op_type for node in model.graph.node],
    }