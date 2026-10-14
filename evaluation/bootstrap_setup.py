#!/usr/bin/env python3
"""Generate reusable EZKL setup only: no witness, proof or resource profiles.

Uses the runtime splitter, setup-key identity and backend stage methods. Each
partition runs in a fresh subprocess so key-generation memory is released.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('model.onnx','settings.json','network.compiled','pk.json','vk.json')

def setup_stages(prover):
    # Do not call run_all(setup_only=True): that currently also makes a witness.
    timings = {}
    for name in ('calibrate_settings_with_fallback','compile_circuit','get_srs','gen_keys'):
        start=time.monotonic(); getattr(prover,name)()
        timings[name]=time.monotonic()-start
    return timings

def generate_job(spec):
    from zkinfer.backends.ezkl.ezkl_prover import EZKLProofStages
    prover=EZKLProofStages(
        input_data_path=spec['input'], onnx_model_path=spec['model'],
        artifact_dir=spec['logs'], proving_cache_enabled=True,
        proving_cache_overwrite=False, proving_cache_type='filesystem',
        proving_cache_root_dir=spec['setup'], local_tmp_dir=spec['tmp'],
    )
    timings=setup_stages(prover)
    directory=Path(spec['setup'])
    if not all((directory/name).is_file() and (directory/name).stat().st_size for name in REQUIRED):
        raise RuntimeError(f'Incomplete setup bundle: {directory}')
    (Path(spec['logs'])/'setup_times.json').write_text(json.dumps(timings,indent=2)+'\n')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload'); p.add_argument('--partition',type=int,default=1)
    p.add_argument('--output',type=Path); p.add_argument('--job',type=Path)
    a=p.parse_args(argv)
    sys.path.insert(0,str(ROOT))
    logging.basicConfig(level=logging.INFO)
    if a.job:
        generate_job(json.loads(a.job.read_text())); return 0
    if not a.workload or a.output is None or a.partition<0:
        p.error('workload, output and nonnegative partition are required')
    from omegaconf import OmegaConf
    from zkinfer.runtime.request_builder import RequestBuilder, EZKL_SETUP_IDENTITY
    from zkinfer.storage.shared_store import SharedStorage
    cfg=OmegaConf.load(ROOT/'experiments/config/workload'/f'{a.workload}.yaml')
    shapes=cfg.get('input_shapes')
    request=SimpleNamespace(request_id='setup-only',name=cfg.name,
        onnx_model_path=cfg.onnx_file,input_data_path=cfg.input_file,
        split_mode='none' if a.partition==0 else 'fixed',ops_per_chunk=max(1,a.partition),
        simplify_model=False,input_shapes=OmegaConf.to_container(shapes,resolve=True) if shapes is not None else None,
        preparation_metrics={})
    a.output.mkdir(parents=True,exist_ok=True)
    storage=SharedStorage('filesystem',str(a.output/'setup'),str(a.output/'inputs'))
    partitions=RequestBuilder(logging.getLogger('setup-only'))._load_or_split_model(request)
    entries=[]; completed=set()
    for index,partition in enumerate(partitions):
        model_bytes=partition.model.SerializeToString()
        key=storage.setup_key_for_model(model_bytes,EZKL_SETUP_IDENTITY)
        model=storage.put_model(model_bytes,setup_key=key)
        input_ref=storage.put_input('bootstrap',str(index),partition.input_data)
        entries.append({'partition':index,'setup_key':key})
        if key in completed: continue
        logs=a.output/'jobs'/key;logs.mkdir(parents=True,exist_ok=True)
        setup=a.output/'setup'/key
        if (logs/'setup_times.json').is_file() and all((setup/name).is_file() and (setup/name).stat().st_size for name in REQUIRED):
            print(f'SKIP completed setup {key}',flush=True)
            completed.add(key); continue
        # A failed key generation can leave partial files. Rebuild only this
        # uncompleted bundle; preserve successfully checkpointed partitions.
        (logs/'setup_times.json').unlink(missing_ok=True)
        for name in REQUIRED:
            if name != 'model.onnx': (setup/name).unlink(missing_ok=True)
        spec={'model':model.uri,'input':input_ref.uri,'setup':str(a.output/'setup'/key),
              'logs':str(logs),'tmp':str(logs/'tmp')}
        path=logs/'setup_job.json';path.write_text(json.dumps(spec,indent=2)+'\n')
        print(f'SETUP ONLY {index+1}/{len(partitions)} {key}',flush=True)
        with (logs/'setup.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        completed.add(key)
    if not entries: raise ValueError('No partitions produced')
    (a.output/'manifest.json').write_text(json.dumps({'setup_only':True,'proofs_generated':0,
        'workload':a.workload,'partition':a.partition,'jobs':entries},indent=2)+'\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
