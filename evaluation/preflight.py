#!/usr/bin/env python3
"""Validate local runtime inputs and freeze an environment record before long runs."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for name in ('ezkl','grpc','hydra','omegaconf','psutil','zkinfer.proto.zkservice_pb2_grpc'):
        importlib.import_module(name)
    from omegaconf import OmegaConf
    import psutil
    models={}
    for name in ('mnist_gan','mobilenet_v2','nano_gpt_4_layers_64_embd'):
        cfg=OmegaConf.load(ROOT/'experiments/config/workload'/f'{name}.yaml')
        for key in ('onnx_file','input_file'):
            path=Path(cfg[key]); path=path if path.is_absolute() else ROOT/path
            if not path.is_file(): raise SystemExit(f'Missing {key}: {path}')
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
            models[f'{name}/{key}']=h.hexdigest()
    versions={}
    for name in ('ezkl','grpcio','hydra-core','omegaconf','psutil','numpy'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not in package metadata'
    record=dict(python=sys.version,platform=platform.platform(),
                total_memory_gib=psutil.virtual_memory().total/1024**3,
                physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
                versions=versions,models=models,
                threads={k:os.getenv(k) for k in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/'environment.json'
    if dest.exists() and json.loads(dest.read_text()) != record:
        raise SystemExit('Environment/model/thread configuration changed. Use a new EVAL_ROOT; do not mix campaigns.')
    if not dest.exists(): dest.write_text(json.dumps(record,indent=2)+'\n')
    print(f'Preflight OK: {record["physical_cores"]} cores; {record["total_memory_gib"]:.2f} GiB RAM; '
          f'{shutil.disk_usage(a.output).free/1024**3:.1f} GiB free disk. Record custom EZKL wheel hash separately.')
    print('Disk availability is informational: large setup keys and retained proof outputs may need multiple TiB.')

if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    main()
