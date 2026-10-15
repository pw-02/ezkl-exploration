#!/usr/bin/env python3
"""Import setup artifacts only. AWS profiles are always collected on AWS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REQUIRED = ('model.onnx', 'settings.json', 'network.compiled', 'pk.json', 'vk.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def backend_identity():
    ezkl = importlib.import_module('ezkl')
    files = sorted(Path(ezkl.__file__).parent.glob('*.so'))
    if not files: raise ValueError('No EZKL native extension found')
    return {p.name: digest(p) for p in files}


def workload_identity(model):
    from omegaconf import OmegaConf
    config = ROOT/'experiments/config/workload'/f'{model}.yaml'
    cfg = OmegaConf.load(config)
    paths = {'config': config}
    for field in ('onnx_file', 'input_file'):
        path = Path(cfg[field])
        paths[field] = path if path.is_absolute() else ROOT/path
    return {name: digest(path) for name, path in paths.items()}


def expected_jobs(model):
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    import logging
    from zkinfer.runtime.request_builder import RequestBuilder, EZKL_SETUP_IDENTITY
    from zkinfer.storage.shared_store import SharedStorage
    cfg = OmegaConf.load(ROOT/'experiments/config/workload'/f'{model}.yaml')
    shapes = cfg.get('input_shapes')
    req = SimpleNamespace(request_id='setup-import', name=cfg.name,
        onnx_model_path=cfg.onnx_file, input_data_path=cfg.input_file,
        split_mode='fixed', ops_per_chunk=1, simplify_model=False,
        input_shapes=OmegaConf.to_container(shapes, resolve=True) if shapes is not None else None,
        preparation_metrics={})
    parts = RequestBuilder(logging.getLogger('setup-import'))._load_or_split_model(req)
    return [{'partition': i, 'setup_key': SharedStorage.setup_key_for_model(
        part.model.SerializeToString(), EZKL_SETUP_IDENTITY)} for i, part in enumerate(parts)]


def validate_manifest(m):
    if m.get('version') != 1 or m.get('partition') != 1 or not m.get('jobs'):
        raise ValueError('Expected a g1 setup import manifest')
    keys = set()
    for i, job in enumerate(m['jobs']):
        key = job['setup_key']
        if job['partition'] != i or not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Invalid partition identity')
        keys.add(key)
    required = {f'setup/{k}/{name}' for k in keys for name in REQUIRED}
    files = m.get('files', {})
    if not required.issubset(files): raise ValueError('Incomplete setup import')
    for rel, entry in files.items():
        allowed = rel in required or re.fullmatch(r'srs/(kzg|ipa)[0-9]+\.srs', rel)
        if not allowed or int(entry['size']) <= 0 or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
            raise ValueError('Invalid import file: '+rel)
    if not any(p.startswith('srs/') for p in files): raise ValueError('Missing SRS')


def manifest_for(bootstrap, model, srs):
    from evaluation.campaign import code_fingerprint
    if not (bootstrap/'DONE.json').is_file(): raise ValueError('Local bootstrap is incomplete')
    done = json.loads((bootstrap/'DONE.json').read_text())
    if done.get('code') != code_fingerprint():
        raise ValueError('Local runtime changed since bootstrap; do not export without reconciling it')
    original = json.loads((bootstrap/'artifacts/manifest.json').read_text())
    if original.get('workload') != model or original.get('partition') != 1 or not original.get('setup_only'):
        raise ValueError('Wrong bootstrap workload or partitioning')
    files = {}
    sources = {}
    for job in original['jobs']:
        key = job['setup_key']
        if not re.fullmatch('[0-9a-f]{64}', key): raise ValueError('Invalid setup key')
        for name in REQUIRED:
            sources[f'setup/{key}/{name}'] = bootstrap/'artifacts/setup'/key/name
        settings = json.loads(sources[f'setup/{key}/settings.json'].read_text())['run_args']
        kind = 'ipa' if str(settings.get('commitment', 'KZG')).lower() == 'ipa' else 'kzg'
        name = f"{kind}{int(settings['logrows'])}.srs"
        sources['srs/'+name] = srs/name
    for rel, path in sources.items():
        if path.is_symlink() or not path.is_file(): raise ValueError('Missing/linked artifact: '+str(path))
        files[rel] = {'size': path.stat().st_size, 'sha256': digest(path)}
    m = dict(version=1, workload=model, partition=1, jobs=original['jobs'], files=files,
             backend=backend_identity(), workload_identity=workload_identity(model),
             runtime_fingerprint=code_fingerprint())
    validate_manifest(m)
    return m, sources


def get_optional(s3, bucket, key):
    try: return json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    except Exception as exc:
        if getattr(exc, 'response', {}).get('Error', {}).get('Code') in {'404','NoSuchKey','NotFound'}: return None
        raise


def publish(s3, bucket, prefix, manifest, sources):
    old = get_optional(s3, bucket, prefix+'/manifest.json')
    if old is not None and old != manifest:
        raise ValueError('Import prefix already identifies different artifacts; choose a new prefix')
    def send(item):
        rel, path = item
        entry = manifest['files'][rel]
        key = prefix+'/'+rel
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in {'404','NoSuchKey','NotFound'}: raise
            head = {}
        if head.get('ContentLength') == entry['size'] and head.get('Metadata',{}).get('sha256') == entry['sha256']:
            return
        s3.upload_file(str(path), bucket, key, ExtraArgs={'Metadata': {'sha256': entry['sha256']}})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send, sources.items()))
    # This marks a complete import, never a completed AWS seed/profile.
    s3.put_object(Bucket=bucket, Key=prefix+'/manifest.json', Body=json.dumps(manifest, sort_keys=True).encode())


def install(s3, bucket, prefix, model, profile_root):
    from evaluation.campaign import code_fingerprint, THREADS, write_json
    m = get_optional(s3, bucket, prefix+'/manifest.json')
    if m is None: raise ValueError('Setup upload incomplete: missing import manifest')
    validate_manifest(m)
    if m['workload'] != model or m['backend'] != backend_identity():
        raise ValueError('Imported workload or native EZKL build differs from AWS')
    if m['workload_identity'] != workload_identity(model) or m['runtime_fingerprint'] != code_fingerprint():
        raise ValueError('Imported model/input/config or runtime differs from AWS')
    if m['jobs'] != expected_jobs(model): raise ValueError('Imported partition identities differ from AWS')
    parent = profile_root/'bootstrap'/f'{model}-g1'
    output = parent/'artifacts'
    # Validate bytes on every resume, including an already installed bootstrap.
    for rel, entry in m['files'].items():
        target = ROOT/rel if rel.startswith('srs/') else output/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == entry['size'] and digest(target) == entry['sha256']: continue
            raise ValueError('Existing artifact differs from import: '+str(target))
        tmp = target.with_name(target.name+'.importing')
        s3.download_file(bucket, prefix+'/'+rel, str(tmp))
        if tmp.stat().st_size != entry['size'] or digest(tmp) != entry['sha256']:
            tmp.unlink(missing_ok=True)
            raise ValueError('Imported artifact checksum mismatch: '+rel)
        tmp.replace(target)
    for job in m['jobs']:
        settings = json.loads((output/'setup'/job['setup_key']/'settings.json').read_text())['run_args']
        kind = 'ipa' if str(settings.get('commitment','KZG')).lower() == 'ipa' else 'kzg'
        if f"srs/{kind}{int(settings['logrows'])}.srs" not in m['files']:
            raise ValueError('Required SRS not covered by manifest')
    signature = dict(workload=model, partition=1, code=code_fingerprint(),
                     threads={k: os.getenv(k) for k in THREADS})
    record = parent/'bootstrap_plan.json'
    if record.exists() and json.loads(record.read_text()) != signature:
        raise ValueError('Existing bootstrap configuration differs; use a new campaign')
    write_json(output/'manifest.json', dict(setup_only=True, proofs_generated=0,
        workload=model, partition=1, jobs=m['jobs']))
    write_json(parent/'import_provenance.json', dict(bucket=bucket, prefix=prefix, manifest=m))
    write_json(record, signature)
    write_json(parent/'DONE.json', signature)
    print('Imported setup only; AWS warm profiling is still required:', model, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--workload', required=True)
    p.add_argument('--bootstrap', type=Path, required=True)
    p.add_argument('--srs', type=Path, default=ROOT/'srs')
    p.add_argument('--profile')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    c = json.loads(a.config.resolve().read_text())
    prefix = c.get('setup_imports', {}).get(a.workload)
    if not prefix or not prefix.startswith('evaluation/setup-imports/') or '..' in Path(prefix).parts:
        raise ValueError('Configure an evaluation/setup-imports/... prefix for this workload')
    if not a.execute:
        print('PLAN: upload setup and required SRS from', a.bootstrap,
              'to', f"s3://{c['bucket']}/{prefix}", '; no profiles, proofs, or cloud changes')
        return
    m, sources = manifest_for(a.bootstrap.resolve(), a.workload, a.srs.resolve())
    import boto3
    s3 = boto3.Session(profile_name=a.profile, region_name=c['region']).client('s3')
    print('Uploading', len(sources), 'files,', round(sum(x['size'] for x in m['files'].values())/1024**3, 2), 'GiB', flush=True)
    publish(s3, c['bucket'], prefix, m, sources)
    print('Setup import uploaded. Run the separate AWS campaign to collect AWS profiles.')


if __name__ == '__main__': main()
