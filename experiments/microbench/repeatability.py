"""Select, prepare and execute repeated isolated proof jobs.

Run from the repository root with python -m experiments.microbench.repeatability.
Heavy ONNX/EZKL imports are deferred so selection and analysis work without EZKL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "mnist_gan": ("experiments/models/mnist_gan/mnist_gan.onnx",
                  "experiments/models/mnist_gan/input.json"),
    "mobilenet_v2": ("experiments/models/mobilenet/mobilenetv2_050_Opset18.onnx",
                     "experiments/models/mobilenet/input.json"),
    "nano_gpt_4_layers_64_embd": ("experiments/models/nanoGPT/nano_gpt_4_layers_64_embd.onnx",
                                "experiments/models/nanoGPT/input.json"),
}
POLICY = {"setup_identity_version": 1, "backend": "ezkl",
          "settings_generation": "gen_settings_then_calibrate_with_fallback",
          "calibration_target": "resources", "proof_type": "single",
          "disable_selector_compression": False}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def settings_fingerprint(path):
    value = json.loads(Path(path).read_text())
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def pin_proving_settings(settings_path, reference_path, fingerprint_path, output, used_cache):
    """Keep the actual proving configuration fixed, preserving calibration candidates.

    Cold runs still perform calibration. Its candidate may differ, but compilation
    and key generation use the first calibrated configuration. Warm cached settings
    must already match that configuration; changing them would invalidate cached keys.
    """
    settings_path, reference_path = Path(settings_path), Path(reference_path)
    fingerprint_path, output = Path(fingerprint_path), Path(output)
    candidate = json.loads(settings_path.read_text())
    candidate_hash = settings_fingerprint(settings_path)
    write_json(output / "calibration_candidate.json", candidate)
    if not reference_path.exists():
        if used_cache:
            raise ValueError("Warm setup has no reference proving settings")
        if fingerprint_path.exists() and fingerprint_path.read_text().strip() != candidate_hash:
            raise ValueError("Legacy fingerprint has no matching reference settings; start a new measurement directory")
        write_json(reference_path, candidate)
    reference = json.loads(reference_path.read_text())
    reference_hash = settings_fingerprint(reference_path)
    if fingerprint_path.exists() and fingerprint_path.read_text().strip() != reference_hash:
        raise ValueError("Reference proving settings were modified")
    changed = candidate_hash != reference_hash
    record = {"candidate_fingerprint": candidate_hash, "reference_fingerprint": reference_hash,
              "candidate_differs": changed, "used_cache": used_cache,
              "policy": "compile and prove using first calibrated settings"}
    write_json(output / "calibration_record.json", record)
    write_json(output / "settings_snapshot.json", reference)
    if used_cache and changed:
        raise ValueError("Warm cached settings differ from the reference; cached circuit/keys cannot be reused")
    if not used_cache:
        write_json(settings_path, reference)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(reference_hash + "\n")
    return reference


def setup_key(model_bytes):
    value = {"model_sha256": digest(model_bytes), "setup_identity": POLICY}
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def partition_number(row):
    return int(row["job_name"].rsplit("_sub_model_", 1)[1])


def select_jobs(rows):
    if not rows or any(r["job_status"] != "COMPLETED" for r in rows):
        raise ValueError("Selection requires a nonempty, fully completed job report")
    if len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate job IDs in report")
    times = [float(r["job_runtime(s)"]) for r in rows]
    memory = [float(r["max_process_memory(GB)"]) for r in rows]
    if not all(math.isfinite(x) and x > 0 for x in times + memory):
        raise ValueError("Invalid duration or memory in report")
    chosen = {}
    for role, q in (("median_duration", .5), ("p90_duration", .9), ("maximum_memory", None)):
        target = percentile(times, q) if q is not None else max(memory)
        distance = [abs(float(r["job_runtime(s)"]) - target) for r in rows] if q is not None else [abs(m-target) for m in memory]
        nearest = min(distance)
        candidates = [r for r, d in zip(rows, distance) if math.isclose(d, nearest, abs_tol=1e-9, rel_tol=0)]
        row = min(candidates, key=partition_number)
        number = partition_number(row)
        if number not in chosen:
            chosen[number] = {"partition": number, "source_job_id": row["job_id"],
                              "setup_key": row["setup_key"], "criteria": [],
                              "baseline_duration_s": float(row["job_runtime(s)"]),
                              "baseline_peak_gib": float(row["max_process_memory(GB)"])}
        chosen[number]["criteria"].append(role)
    return list(chosen.values())


def reports(source):
    if source.is_dir():
        return [(str(p), p.read_text()) for p in sorted(source.rglob("job_report.csv"))]
    with tarfile.open(source) as archive:
        return [(m.name, archive.extractfile(m).read().decode("utf-8-sig"))
                for m in archive.getmembers() if m.isfile() and m.name.endswith("/reports/job_report.csv")]


def selection(source, models):
    found = {}
    for name, content in reports(source):
        rows = list(csv.DictReader(io.StringIO(content)))
        if not rows:
            continue
        model = rows[0]["job_name"].rsplit("_sub_model_", 1)[0]
        if model not in models:
            continue
        if model in found:
            raise ValueError(f"Multiple reports for {model}; select a narrower source")
        found[model] = {"model": model, "source_report": name, "jobs": select_jobs(rows)}
    if set(found) != set(models):
        raise ValueError(f"Missing reports for {set(models) - set(found)}")
    return [found[name] for name in models]


def fresh_output(path):
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}; use a new directory")
    path.mkdir(parents=True)
    return path


def generate_inputs(session, baseline, output, repetitions, seed, noise_scale, integer_bound=None):
    """Explicit synthetic pilot mode; real input files are preferred."""
    import numpy as np
    from zkinfer.utils.onnx import load_or_generate_inputs
    original = load_or_generate_inputs(session, str(baseline))
    rng = np.random.default_rng(seed)
    result = []
    for k in range(1, repetitions + 1):
        arrays = []
        for item in session.get_inputs():
            value = original[item.name]
            if np.issubdtype(value.dtype, np.floating):
                scale = max(float(np.sqrt(np.mean(value.astype(float) ** 2))), 1e-3)
                varied = (value + rng.normal(0, noise_scale * scale, value.shape)).astype(value.dtype)
            elif item.name in {"attention_mask", "token_type_ids"} or value.dtype == np.bool_:
                varied = value.copy()
            elif integer_bound is not None and np.issubdtype(value.dtype, np.integer):
                varied = rng.integers(0, integer_bound, size=value.shape).astype(value.dtype)
            else:
                raise ValueError("Integer inputs require real input files or an explicit --integer-upper-bound")
            if not np.isfinite(varied).all():
                raise ValueError("Synthetic input contains nonfinite values")
            arrays.append(varied.flatten().tolist())
        path = output / f"input_{k:03d}.json"
        write_json(path, {"input_data": arrays})
        result.append(path)
    return result


def prepare(args):
    import onnxruntime as ort
    from zkinfer.graph.onnx_splitter import split_onnx_model_with_inputs
    selected = selection(args.source, args.models)
    config = json.loads(args.config.read_text()) if args.config else {}
    output = fresh_output(args.output)
    write_json(output / "selection.json", selected)
    manifest = {"version": 1, "repetitions": args.repetitions, "seed": args.seed,
                "input_mode": args.input_mode, "noise_scale": args.noise_scale,
                "setup_policy": POLICY, "cases": []}
    for group in selected:
        model_name = group["model"]
        entry = config.get(model_name, {})
        model_path = (ROOT / entry.get("model", MODELS[model_name][0])).resolve()
        baseline = (ROOT / entry.get("baseline_input", MODELS[model_name][1])).resolve()
        if not model_path.is_file() or not baseline.is_file():
            raise FileNotFoundError(f"Missing model/baseline for {model_name}: {model_path}, {baseline}")
        parent_dir = output / "parent_inputs" / model_name
        parent_dir.mkdir(parents=True)
        shutil.copy2(baseline, parent_dir / "input_000.json")
        if args.input_mode == "files":
            inputs = [(ROOT / p).resolve() for p in entry.get("inputs", [])]
            if len(inputs) < args.repetitions:
                raise ValueError(f"{model_name}: provide {args.repetitions} input JSON paths in --config, or use --input-mode perturb for a synthetic pilot")
            inputs = inputs[:args.repetitions]
        else:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            inputs = generate_inputs(session, baseline, parent_dir, args.repetitions,
                                     args.seed, args.noise_scale, args.integer_upper_bound)
        full_inputs = [parent_dir / "input_000.json"]
        hashes = set()
        for k, src in enumerate(inputs, 1):
            target = parent_dir / f"input_{k:03d}.json"
            if src.resolve() != target.resolve():
                shutil.copy2(src, target)
            value_hash = digest(json.dumps(json.loads(target.read_text()), sort_keys=True).encode())
            if value_hash in hashes:
                raise ValueError(f"Repeated full-model input for {model_name}, repetition {k}")
            hashes.add(value_hash)
            full_inputs.append(target)
        expected = {}
        cases = {}
        for chosen in group["jobs"]:
            number = chosen["partition"]
            case_dir = output / "circuits" / model_name / f"partition_{number}"
            case_dir.mkdir(parents=True)
            cases[number] = {**chosen, "model": model_name,
                             "directory": str(case_dir.relative_to(output)), "inputs": []}
        for k, input_path in enumerate(full_inputs):
            print(f"Materializing {model_name} input {k}/{args.repetitions}", flush=True)
            partitions = split_onnx_model_with_inputs(
                model_path=str(model_path), input_data_path=str(input_path),
                split_mode="fixed", split_group_size=1, model_name=model_name)
            for number, case in cases.items():
                part = next((p for p in partitions if p.name == f"sub_model_{number}" or
                             p.name.endswith(f"_sub_model_{number}")), None)
                if part is None:
                    raise ValueError(f"Partition {number} not found for {model_name}")
                model_bytes = part.model.SerializeToString()
                case_dir = output / case["directory"]
                if k == 0:
                    if setup_key(model_bytes) != case["setup_key"]:
                        raise ValueError(f"{model_name} partition {number}: circuit does not match the source trace. Check model, splitter, and setup policy versions.")
                    expected[number] = digest(model_bytes)
                    (case_dir / "model.onnx").write_bytes(model_bytes)
                    case["model_sha256"] = expected[number]
                elif digest(model_bytes) != expected[number]:
                    raise ValueError(f"Circuit changed for {model_name} partition {number} on input {k}")
                target = case_dir / f"input_{k:03d}.json"
                write_json(target, part.input_data)
                case["inputs"].append(target.name)
        manifest["cases"].extend(cases.values())
        for case in cases.values():
            case_dir = output / case["directory"]
            case["input_sha256"] = [digest((case_dir / name).read_bytes()) for name in case["inputs"]]
            varied = [digest(json.dumps(json.loads((case_dir / name).read_text()), sort_keys=True).encode())
                      for name in case["inputs"][1:]]
            if len(set(varied)) != args.repetitions:
                raise ValueError(f"{model_name} partition {case['partition']}: boundary inputs do not vary across every repetition. Supply a more varied full-model input set.")
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(manifest['cases'])} circuits in {output}")


def worker(task_path):
    import logging
    from zkinfer.runtime.node.prover_task import (
        LocalProverTask,
        ProverTaskRunner,
        StageAwareEZKLProofStages,
    )
    from zkinfer.runtime.node.stage_admission import StageAdmissionController

    payload = json.loads(task_path.read_text())
    task = LocalProverTask.from_dict(payload["task"])
    output = Path(task.output_dir)
    logging.basicConfig(level=logging.INFO)
    fingerprint_path = Path(payload["fingerprint_path"])

    class Stages(StageAwareEZKLProofStages):
        def calibrate_settings_with_fallback(self):
            request_input = self.input_data_path
            try:
                self.input_data_path = payload["calibration_input"]
                result = super().calibrate_settings_with_fallback()
            finally:
                self.input_data_path = request_input
            self.ezkl_settings_dict = pin_proving_settings(
                self.settings_path, fingerprint_path.with_name("reference_settings.json"),
                fingerprint_path, output, used_cache=bool(result[0]))
            return result

    class Runner(ProverTaskRunner):
        def _start_resource_logger(self, task, status_file):
            process, path = super()._start_resource_logger(task, status_file)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and process.poll() is None:
                if Path(path).is_file() and ",process_tree," in Path(path).read_text():
                    return process, path
                time.sleep(.01)
            self._stop_process(process)
            raise RuntimeError("Resource sampler did not become ready before proof execution")

        def _build_proof_stages(self, task, status_file):
            return Stages(
                input_data_path=task.input_path,
                onnx_model_path=task.model_path,
                artifact_dir=task.output_dir,
                proving_cache_enabled=True,
                proving_cache_overwrite=False,
                proving_cache_type="filesystem",
                proving_cache_root_dir=task.setup_dir,
                local_tmp_dir=task.tmp_dir,
                status_file=status_file,
                srs_dir=payload["srs_dir"],
                admission=StageAdmissionController(
                    root_dir=task.stage_admission_dir,
                    setup_key=task.setup_key,
                    enabled=False,
                    fallback_memory_gb=task.predicted_peak_memory_gb,
                    memory_safety_reserve_gb=task.memory_safety_reserve_gb,
                    history_percentile=task.history_percentile,
                ),
            )

    started = time.perf_counter()
    try:
        result = Runner().run(task)
        proof_wall_time = time.perf_counter() - started
        expected_hit = payload["condition"] == "warm"
        cache_flags = [bool(result.metrics.get(k)) for k in (
            "ezkl_calibrate_settings_used_cache", "ezkl_compile_circuit_used_cache", "ezkl_key_gen_used_cache")]
        if cache_flags != [expected_hit] * 3:
            raise RuntimeError(f"Unexpected setup cache flags: {cache_flags}")
        if not expected_hit and not result.metrics.get("ezkl_fused_stage_trace_available"):
            raise RuntimeError("Detailed native stage timestamps are unavailable; rebuild the custom EZKL wheel")
        verification = {"enabled": payload.get("verify", False), "valid": None, "time_s": None}
        if verification["enabled"]:
            import ezkl
            verify_started = time.perf_counter()
            settings = json.loads((output / "settings_snapshot.json").read_text())
            args = settings["run_args"]
            prefix = "ipa" if str(args.get("commitment", "KZG")).lower() == "ipa" else "kzg"
            srs = Path(payload["srs_dir"]) / f"{prefix}{args['logrows']}.srs"
            verification["valid"] = bool(ezkl.verify(
                proof_path=str(Path(task.tmp_dir) / "proof.pf"),
                settings_path=str(Path(task.setup_dir) / "settings.json"),
                vk_path=str(Path(task.setup_dir) / "vk.json"), srs_path=str(srs)))
            verification["time_s"] = time.perf_counter() - verify_started
            if not verification["valid"]:
                raise RuntimeError("Proof verification failed")
        write_json(output / "result.json", {"ok": True, "wall_time_s": proof_wall_time,
                   "verification": verification,
                   "settings_fingerprint": fingerprint_path.read_text().strip(),
                   "result": result.to_dict()})
        return 0
    except Exception as exc:
        logging.exception("Microbenchmark proof failed")
        write_json(output / "result.json", {"ok": False, "error": str(exc),
                   "wall_time_s": time.perf_counter()-started})
        return 1


def metadata(args, manifest):
    def git(*command):
        try:
            return subprocess.check_output(["git", *command], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import importlib.metadata
    import psutil
    import ezkl.ezkl
    native = Path(ezkl.ezkl.__file__)
    h = hashlib.sha256()
    with native.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {"commit": git("rev-parse", "HEAD"), "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
            "python": sys.version, "platform": platform.platform(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "ezkl_version": importlib.metadata.version("ezkl"), "ezkl_native_sha256": h.hexdigest(),
            "prepared_manifest_sha256": digest((args.prepared / "manifest.json").read_bytes()),
            "sampling_interval_s": args.interval, "repetitions": manifest["repetitions"],
            "verification_enabled": args.verify,
            "setup_conditions": args.conditions,
            "calibration_policy": "same reference input; actual proving settings pinned to first calibration",
            "srs_dir": str(args.srs_dir.resolve()), "setup_scope": "filesystem artifacts; OS page cache not flushed"}


def run(args):
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    output = fresh_output(args.output)
    write_json(output / "manifest.json", manifest)
    write_json(output / "run_metadata.json", metadata(args, manifest))
    selected = [c for c in manifest["cases"] if not args.case or
                f"{c['model']}:{c['partition']}" in args.case]
    if not selected or (args.case and len(selected) != len(set(args.case))):
        raise ValueError("Unknown --case; use model:partition from the prepared manifest")
    for case in selected:
        source = args.prepared / case["directory"]
        if digest((source / "model.onnx").read_bytes()) != case["model_sha256"]:
            raise ValueError("Prepared model changed")
        case_root = output / case["model"] / f"partition_{case['partition']}"
        for condition in args.conditions:
            group = case_root / condition
            fingerprint = case_root / "settings.fingerprint"
            cache = group / "setup-cache"
            repetitions = ([0] if condition == "warm" else []) + list(range(1, manifest["repetitions"]+1))
            try:
                for k in repetitions:
                    warmup = k == 0
                    path = group / ("warmup" if warmup else f"run_{k:03d}")
                    path.mkdir(parents=True)
                    temp = path / "temporary"
                    setup = cache if condition == "warm" else temp / "setup"
                    input_path = source / case["inputs"][k]
                    if digest(input_path.read_bytes()) != case["input_sha256"][k]:
                        raise ValueError("Prepared input changed")
                    task = {
                        "node_id": "microbench",
                        "request_id": "repeatability",
                        "job_id": f"partition_{case['partition']}",
                        "setup_key": case["setup_key"],
                        "model_path": str((source / "model.onnx").resolve()),
                        "input_path": str(input_path.resolve()),
                        "setup_dir": str(setup.resolve()),
                        "tmp_dir": str((temp / "work").resolve()),
                        "output_dir": str(path.resolve()),
                        "output_prefix": str((temp / "published").resolve()),
                        "output_backend": "filesystem",
                        "resource_sampling_interval_sec": args.interval,
                        "stage_admission_enabled": False,
                        "stage_admission_dir": str((temp / "admission").resolve()),
                        "keep_proof_artifacts": False,
                    }
                    write_json(path / "execution.json", {"model": case["model"], "partition": case["partition"],
                               "condition": condition, "repetition": k, "warmup": warmup,
                               "setup_key": case["setup_key"], "input_sha256": digest(input_path.read_bytes())})
                    write_json(path / "task.json", {"task": task, "condition": "cold" if warmup else condition,
                               "fingerprint_path": str(fingerprint.resolve()),
                               "calibration_input": str((source / case["inputs"][0]).resolve()),
                               "srs_dir": str(args.srs_dir.resolve()), "verify": args.verify})
                    print(f"{case['model']} partition {case['partition']} {condition} {'warmup' if warmup else k}", flush=True)
                    with (path / "prover.log").open("w") as log:
                        process = subprocess.Popen([sys.executable, "-m", __name__ if __name__ != "__main__" else
                            "experiments.microbench.repeatability", "_worker", "--task", str((path / "task.json").resolve())],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            code = process.wait()
                        except KeyboardInterrupt:
                            import signal
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait()
                            raise
                    shutil.rmtree(temp, ignore_errors=True)
                    if code:
                        raise RuntimeError(f"Proof failed; see {path / 'prover.log'}")
            finally:
                if not args.keep_setup:
                    shutil.rmtree(cache, ignore_errors=True)
    print(f"Finished. Analyze with: python -m experiments.microbench.analysis --runs {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("select")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--input-mode", choices=["files", "perturb"], default="files")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--noise-scale", type=float, default=.01)
    p.add_argument("--integer-upper-bound", type=int)
    p = commands.add_parser("run")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=.1)
    p.add_argument("--conditions", nargs="+", choices=["cold", "warm"], default=["cold", "warm"])
    p.add_argument("--case", action="append")
    p.add_argument("--srs-dir", type=Path, default=ROOT / "srs")
    p.add_argument("--keep-setup", action="store_true")
    p.add_argument("--verify", action="store_true", help="Verify each proof after profiling ends")
    p = commands.add_parser("_worker")
    p.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "repetitions") and args.repetitions < 3:
        parser.error("At least three repetitions are needed for causal prediction")
    if hasattr(args, "interval") and (not math.isfinite(args.interval) or args.interval <= 0):
        parser.error("Sampling interval must be positive")
    if hasattr(args, "noise_scale") and (not math.isfinite(args.noise_scale) or args.noise_scale <= 0):
        parser.error("Noise scale must be positive")
    if getattr(args, "integer_upper_bound", None) is not None and args.integer_upper_bound <= 1:
        parser.error("Integer upper bound must exceed one")
    if hasattr(args, "conditions"):
        args.conditions = list(dict.fromkeys(args.conditions))
    if args.command == "select":
        write_json(args.output, selection(args.source, args.models))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        raise SystemExit(worker(args.task))


if __name__ == "__main__":
    main()
