# experiments/model_stats_report.py

import csv
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import ezkl
import onnx
from onnx import numpy_helper

from zkinfer.storage.io import compute_bytes_md5_hex


DEFAULT_MODELS_DIR = "experiments/models"
DEFAULT_OUTPUT_CSV = "analysis/model_stats_report.csv"
DEFAULT_TMP_DIR = "_ezkl_model_stats_tmp"

DEFAULT_RUN_CALIBRATION = True
DEFAULT_CLEAN_TMP = True
DEFAULT_APPEND_TO_EXISTING_REPORT = True


BASE_FIELDNAMES = [
    "model_name",
    "model_dir",
    "onnx_model_path",
    "onnx_file_size_bytes",
    "onnx_file_size_mb",
    "input_data_path",
    "has_input_data",
    "overall_ok",
    "failure_stage",
    "failure_error",
    "onnx_stats_ok",
    "onnx_stats_error",
    "ezkl_gen_settings_ok",
    "ezkl_calibration_ok",
    "ezkl_error",
]


def make_run_args() -> ezkl.PyRunArgs:
    run_args = ezkl.PyRunArgs()

    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"

    return run_args


def run_args_to_dict(run_args: ezkl.PyRunArgs) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    for key in dir(run_args):
        if key.startswith("_"):
            continue

        try:
            value = getattr(run_args, key)
        except Exception:
            continue

        if callable(value):
            continue

        if isinstance(value, (list, tuple, dict)):
            info[f"run_args.{key}"] = json.dumps(value)
        else:
            info[f"run_args.{key}"] = value

    return info


def flatten_dict(value: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(item, dict):
            out.update(flatten_dict(item, name))
        elif isinstance(item, list):
            out[name] = json.dumps(item)
        else:
            out[name] = item

    return out


def discover_onnx_models(models_dir: Path) -> list[Path]:
    return sorted(path for path in models_dir.rglob("*.onnx") if path.is_file())


def find_input_for_model(model_path: Path) -> Optional[Path]:
    candidates = [
        model_path.parent / "input.json",
        model_path.with_suffix(".json"),
        model_path.parent / f"{model_path.stem}_input.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_model_name(models_dir: Path, model_path: Path) -> str:
    rel = model_path.relative_to(models_dir)
    return str(rel.with_suffix("")).replace(os.sep, "__")


def tensor_shape_from_value_info(value_info) -> str:
    tensor_type = value_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        return ""

    dims = []

    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    return json.dumps(dims)


def onnx_model_stats(model_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(model_path))
    graph = model.graph

    model_bytes = model.SerializeToString()
    model_hash = compute_bytes_md5_hex(model_bytes)

    nodes = list(graph.node)
    initializers = list(graph.initializer)

    op_counts = Counter(node.op_type for node in nodes)

    initializer_numel = 0
    initializer_bytes = 0
    largest_initializer_name = ""
    largest_initializer_bytes = 0

    for initializer in initializers:
        array = numpy_helper.to_array(initializer)
        nbytes = int(array.nbytes)

        initializer_numel += int(array.size)
        initializer_bytes += nbytes

        if nbytes > largest_initializer_bytes:
            largest_initializer_bytes = nbytes
            largest_initializer_name = initializer.name

    initializer_names = {initializer.name for initializer in initializers}
    runtime_inputs = [
        graph_input
        for graph_input in graph.input
        if graph_input.name not in initializer_names
    ]

    runtime_input_shapes = {
        graph_input.name: tensor_shape_from_value_info(graph_input)
        for graph_input in runtime_inputs
    }

    output_shapes = {
        graph_output.name: tensor_shape_from_value_info(graph_output)
        for graph_output in graph.output
    }

    has_dynamic_shapes = any(
        "?" in shape
        for shape in list(runtime_input_shapes.values()) + list(output_shapes.values())
    )

    return {
        "model_hash": model_hash,
        "onnx_ir_version": model.ir_version,
        "onnx_opset_imports": json.dumps(
            {
                opset.domain or "ai.onnx": opset.version
                for opset in model.opset_import
            }
        ),
        "num_nodes": len(nodes),
        "num_initializers": len(initializers),
        "num_parameters": initializer_numel,
        "initializer_size_bytes": initializer_bytes,
        "initializer_size_mb": round(initializer_bytes / 1024 / 1024, 4),
        "largest_initializer_name": largest_initializer_name,
        "largest_initializer_size_bytes": largest_initializer_bytes,
        "largest_initializer_size_mb": round(largest_initializer_bytes / 1024 / 1024, 4),
        "num_graph_inputs": len(graph.input),
        "num_runtime_inputs": len(runtime_inputs),
        "num_outputs": len(graph.output),
        "runtime_input_names": json.dumps([x.name for x in runtime_inputs]),
        "runtime_input_shapes": json.dumps(runtime_input_shapes),
        "output_names": json.dumps([x.name for x in graph.output]),
        "output_shapes": json.dumps(output_shapes),
        "has_dynamic_shapes": has_dynamic_shapes,
        "num_unique_ops": len(op_counts),
        "op_counts": json.dumps(dict(sorted(op_counts.items()))),
        "num_conv_ops": op_counts.get("Conv", 0),
        "num_gemm_ops": op_counts.get("Gemm", 0),
        "num_matmul_ops": op_counts.get("MatMul", 0),
        "num_add_ops": op_counts.get("Add", 0),
        "num_mul_ops": op_counts.get("Mul", 0),
        "num_relu_ops": op_counts.get("Relu", 0),
        "num_softmax_ops": op_counts.get("Softmax", 0),
        "num_layernorm_ops": op_counts.get("LayerNormalization", 0),
        **{
            f"op_count.{op_name}": count
            for op_name, count in sorted(op_counts.items())
        },
    }


def ezkl_settings_stats(
    model_name: str,
    model_path: Path,
    input_path: Optional[Path],
    tmp_dir: Path,
    run_calibration: bool,
) -> Dict[str, Any]:
    safe_model_name = model_name.replace("/", "__").replace("\\", "__")
    settings_path = tmp_dir / f"{safe_model_name}_settings.json"

    info: Dict[str, Any] = {
        "ezkl_settings_path": str(settings_path),
        "ezkl_gen_settings_ok": False,
        "ezkl_calibration_ok": False,
        "ezkl_error": "",
    }

    run_args = make_run_args()
    info.update(run_args_to_dict(run_args))

    try:
        ezkl.gen_settings(
            str(model_path),
            str(settings_path),
            py_run_args=run_args,
        )

        info["ezkl_gen_settings_ok"] = True

        if run_calibration:
            if input_path is None:
                raise FileNotFoundError(
                    f"No input file found for calibration: {model_path}"
                )

            ezkl.calibrate_settings(
                str(input_path),
                str(model_path),
                str(settings_path),
                "resources",
            )

            info["ezkl_calibration_ok"] = True

        with open(settings_path, "r", encoding="utf-8") as file:
            settings = json.load(file)

        info.update(flatten_dict(settings, "ezkl"))

    except Exception as exc:
        info["ezkl_error"] = repr(exc)

    return info


class IncrementalCsvWriter:
    def __init__(
        self,
        output_path: Path,
        base_fieldnames: list[str],
        append: bool = False,
    ) -> None:
        self.output_path = output_path
        self.fieldnames = list(base_fieldnames)
        self.rows: list[Dict[str, Any]] = []
        self.append = append

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.output_path.exists():
            self._load_existing_rows()
        elif self.output_path.exists():
            self.output_path.unlink()

    def _load_existing_rows(self) -> None:
        with open(self.output_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in self.fieldnames:
                        self.fieldnames.append(field)

            for row in reader:
                self.rows.append(dict(row))

    def add_row(self, row: Dict[str, Any]) -> None:
        for key in row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)

        self.rows.append(row)
        self._rewrite()

    def _rewrite(self) -> None:
        with open(self.output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(self.rows)


def process_model(
    models_dir: Path,
    model_path: Path,
    tmp_dir: Path,
    run_calibration: bool,
) -> Dict[str, Any]:
    model_name = build_model_name(models_dir, model_path)
    input_path = find_input_for_model(model_path)

    row: Dict[str, Any] = {
        "model_name": model_name,
        "model_dir": str(model_path.parent),
        "onnx_model_path": str(model_path),
        "onnx_file_size_bytes": model_path.stat().st_size,
        "onnx_file_size_mb": round(model_path.stat().st_size / 1024 / 1024, 4),
        "input_data_path": str(input_path) if input_path else "",
        "has_input_data": input_path is not None,
        "overall_ok": True,
        "failure_stage": "",
        "failure_error": "",
    }

    try:
        row.update(onnx_model_stats(model_path))
        row["onnx_stats_ok"] = True
        row["onnx_stats_error"] = ""
    except Exception as exc:
        row["overall_ok"] = False
        row["failure_stage"] = "onnx_stats"
        row["failure_error"] = repr(exc)
        row["onnx_stats_ok"] = False
        row["onnx_stats_error"] = repr(exc)
        logging.exception("ONNX stats failed for %s", model_path)

    ezkl_info = ezkl_settings_stats(
        model_name=model_name,
        model_path=model_path,
        input_path=input_path,
        tmp_dir=tmp_dir,
        run_calibration=run_calibration,
    )

    row.update(ezkl_info)

    if ezkl_info.get("ezkl_error"):
        row["overall_ok"] = False

        if not row["failure_stage"]:
            row["failure_stage"] = "ezkl_settings"
            row["failure_error"] = ezkl_info["ezkl_error"]

        logging.error(
            "EZKL settings failed for %s: %s",
            model_path,
            ezkl_info["ezkl_error"],
        )

    return row


def generate_report(
    models_dir: Path,
    output_path: Path,
    tmp_dir: Path,
    run_calibration: bool,
    clean_tmp: bool,
    append_to_existing_report: bool,
) -> None:
    if clean_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    models = discover_onnx_models(models_dir)
    writer = IncrementalCsvWriter(
        output_path=output_path,
        base_fieldnames=BASE_FIELDNAMES,
        append=append_to_existing_report,
    )

    total = 0
    success_count = 0
    failure_count = 0

    logging.info("Found %d ONNX models under %s", len(models), models_dir)
    logging.info(
        "Report mode: %s",
        "append" if append_to_existing_report else "overwrite",
    )

    for model_path in models:
        total += 1
        logging.info("Processing %s", model_path)

        row = process_model(
            models_dir=models_dir,
            model_path=model_path,
            tmp_dir=tmp_dir,
            run_calibration=run_calibration,
        )

        if row.get("overall_ok"):
            success_count += 1
            logging.info("Completed %s", model_path)
        else:
            failure_count += 1
            logging.warning(
                "Completed with failure %s | stage=%s | error=%s",
                model_path,
                row.get("failure_stage"),
                row.get("failure_error"),
            )

        writer.add_row(row)

        logging.info(
            "Progress: total=%d success=%d failed=%d report=%s",
            total,
            success_count,
            failure_count,
            output_path,
        )

    logging.info(
        "Done. total=%d success=%d failed=%d report=%s",
        total,
        success_count,
        failure_count,
        output_path,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    generate_report(
        models_dir=Path(DEFAULT_MODELS_DIR),
        output_path=Path(DEFAULT_OUTPUT_CSV),
        tmp_dir=Path(DEFAULT_TMP_DIR),
        run_calibration=DEFAULT_RUN_CALIBRATION,
        clean_tmp=DEFAULT_CLEAN_TMP,
        append_to_existing_report=DEFAULT_APPEND_TO_EXISTING_REPORT,
    )


if __name__ == "__main__":
    main()