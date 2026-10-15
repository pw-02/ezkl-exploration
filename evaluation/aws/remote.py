#!/usr/bin/env python3
"""Executed as ubuntu through SSM. Each action belongs to one campaign/attempt."""
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def upload(s3, bucket, directory, prefix):
    for p in sorted(directory.rglob('*')):
        if p.is_file():
            s3.upload_file(str(p), bucket, prefix + '/' + p.relative_to(directory).as_posix())


def download(s3, bucket, prefix, directory):
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix + '/'):
        for item in page.get('Contents', []):
            rel = item['Key'][len(prefix) + 1:]
            if not rel or '..' in Path(rel).parts or Path(rel).is_absolute():
                raise ValueError('Invalid S3 relative path')
            dest = directory / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, item['Key'], str(dest))


def unit_name(c, attempt, role):
    key = hashlib.sha256((c['campaign'] + '/' + attempt).encode()).hexdigest()[:16]
    return 'zkinfer-eval-' + key + '-' + role


def wait_for_coordinator(unit, port, log, timeout=60):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        state = subprocess.run(['sudo','systemctl','is-active',unit],
                               text=True,capture_output=True)
        if state.stdout.strip() not in {'active','activating'}:
            detail = log.read_text(errors='replace')[-12000:] if log.exists() else 'No coordinator log'
            raise RuntimeError('Coordinator exited before readiness:\n'+detail)
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=0.5):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError('Coordinator did not listen on port '+str(port))


def main(payload):
    import boto3
    import psutil
    from omegaconf import OmegaConf
    c, action = payload['config'], payload['action']
    os.chdir(ROOT)
    os.environ.update(RAYON_NUM_THREADS=str(c['threads']), OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', AWS_DEFAULT_REGION=c['region'])
    s3 = boto3.client('s3', region_name=c['region'])
    bucket = c['bucket']
    prefix = 'evaluation/' + c['campaign']
    root = ROOT / 'evaluation/results' / c['campaign']
    root.mkdir(parents=True, exist_ok=True)
    if action == 'doctor':
        import ezkl
        import grpc
        with socket.socket() as probe:
            probe.bind(('0.0.0.0',c['port']))
        hashes = {}
        for model in c['workloads']:
            cfg = OmegaConf.load(ROOT/'experiments/config/workload'/f'{model}.yaml')
            for field in ('onnx_file', 'input_file'):
                path = Path(cfg[field])
                if not path.is_absolute(): path = ROOT/path
                h = hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
                hashes[model+'/'+field] = h.hexdigest()
        total = psutil.virtual_memory().total/1024**3
        if total-c['memory_budget_gib'] < 8:
            raise ValueError('Budget must leave at least 8 GiB actual RAM')
        key = prefix+'/checks/'+payload['instance_id']+'.json'
        info = dict(memory_gib=total, cpus=psutil.cpu_count(), models=hashes,
                    python=sys.version, ezkl=importlib.metadata.version('ezkl'),
                    free_disk_gib=shutil.disk_usage(root).free/1024**3,
                    source_sha256=payload['source_sha256'])
        backend_hash = hashlib.sha256()
        for path in sorted(Path(ezkl.__file__).parent.glob('*.so')):
            backend_hash.update(path.name.encode())
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''): backend_hash.update(block)
        info['ezkl_native_sha256'] = backend_hash.hexdigest()
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(info).encode())
        assert json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read()) == info
        print(json.dumps(info)); return
    if action == 'prepare':
        model = payload['workload']
        seed = prefix+'/seeds/'+model
        try:
            s3.head_object(Bucket=bucket,Key=seed+'/DONE.json')
            print('SKIP completed AWS preparation',model); return
        except Exception as exc:
            if getattr(exc,'response',{}).get('Error',{}).get('Code') not in {'404','NoSuchKey','NotFound'}: raise
        profile_root = root/'preparation'
        import_prefix = c.get('setup_imports', {}).get(model)
        if import_prefix:
            from evaluation.aws.setup_import import install
            install(s3, bucket, import_prefix, model, profile_root)
        command = [sys.executable, 'evaluation/campaign.py', 'prepare', '--root', str(profile_root),
                   '--id', model+'-g1-warm', '--workload', model, '--partition', '1',
                   '--condition', 'warm', '--reserve', '8', '--repetitions', '1',
                   '--timeout', str(c['timeout_seconds']), '--cache-gib', str(c['cache_gib']),
                   '--sampling', str(c['sampling_seconds']), '--execute', '--discard-copies']
        from evaluation.campaign import launch
        launch(command, timeout=c['timeout_seconds'])
        parent = profile_root/'profiles'/f'{model}-g1-warm'
        done = json.loads((parent/'DONE.json').read_text())
        task = parent/done['attempt']
        peaks = [float(o.get('peak_memory_gb',0))
                 for p in (task/'snapshot/coordinator').glob('*/resource_profile.json')
                 for o in json.loads(p.read_text()).get('_resource_history',{}).get('observations',[])]
        if not peaks or max(peaks)>c['memory_budget_gib']:
            raise ValueError('Missing profiles or a job exceeds the per-node proof-memory budget')
        from evaluation.campaign import setup_source
        upload(s3, bucket, setup_source(task), seed+'/setup')
        upload(s3, bucket, task/'snapshot', seed+'/snapshot')
        upload(s3, bucket, ROOT/'srs', seed+'/srs')
        # Reports and frozen profile provenance are retained separately from keys.
        for p in task.rglob('*'):
            if p.is_file() and (p.suffix in {'.csv', '.log'} or p.name in {'metadata.json','plan.json','summary.json','run_status.json'}):
                s3.upload_file(str(p), bucket, seed+'/reports/'+p.relative_to(task).as_posix())
        provenance = profile_root/'bootstrap'/f'{model}-g1'/'import_provenance.json'
        if provenance.is_file():
            s3.upload_file(str(provenance), bucket, seed+'/import_provenance.json')
        s3.put_object(Bucket=bucket, Key=seed+'/DONE.json', Body=json.dumps(done).encode())
        return
    attempt = payload['attempt']
    out = root/'runs'/attempt
    trial_prefix = prefix+'/runs/'+attempt
    if action == 'init':
        out.mkdir(parents=True, exist_ok=False)
        seed = prefix+'/seeds/'+payload['workload']
        s3.head_object(Bucket=bucket, Key=seed+'/DONE.json')
        download(s3, bucket, seed+'/snapshot/node', out/'node-state/resource_profiles')
        if not list((out/'node-state/resource_profiles').glob('*.json')):
            raise ValueError('Missing node profiles')
        download(s3, bucket, seed+'/srs', ROOT/'srs')
        if payload['leader']:
            # Independent writable S3 histories per attempt; shared immutable setup seed.
            download(s3, bucket, seed+'/snapshot/coordinator', out/'initial-coordinator-profiles')
            upload(s3, bucket, out/'initial-coordinator-profiles', trial_prefix+'/profiles')
        write(out/'deployment.json', payload)
        return
    if action == 'start':
        role = payload['role']
        values = {
            'coordinator.host': '0.0.0.0' if role == 'coordinator' else payload['coordinator_ip'],
            'coordinator.port': c['port'], 'coordinator.scheduling_policy': 'resource_aware',
            'coordinator.runs_dir': str(out/'requests'), 'coordinator.logs_dir': str(out/'logs/coordinator'),
            'coordinator.state_db_path': str(out/'coordinator.sqlite3'),
            'node.runs_dir': str(out/'requests'), 'node.logs_dir': str(out/'logs/node'),
            'node.node_id': payload['instance_id'], 'node.runtime_state_dir': str(out/'node-state'),
            'node.local_cache.path': str(out/'cache'), 'node.local_cache.max_gb': c['cache_gib'],
            'node.max_concurrent_jobs': None, 'node.resource_sampling_interval_sec': c['sampling_seconds'],
            'node.stage_admission.enabled': True,
            'node.stage_admission.mode': 'whole_job' if payload['baseline']=='whole-job' else 'stage',
            'node.stage_admission.memory_safety_reserve_gb': psutil.virtual_memory().total/1024**3-c['memory_budget_gib'],
            'node.stage_admission.history_percentile': 0.9,
            'storage.backend': 's3', 'storage.s3_bucket': bucket, 'storage.s3_prefix': prefix,
            'storage.request_data.path': 'runs/'+attempt+'/request-data',
            'storage.reusable_setup.enabled': True,
            'storage.reusable_setup.path': 'seeds/'+payload['workload']+'/setup',
            'storage.resource_profiles.enabled': True,
            'storage.resource_profiles.path': 'runs/'+attempt+'/profiles', 'verification.enabled': True,
        }
        command = [sys.executable, '-u'] + (['evaluation/aws/coordinator.py'] if role=='coordinator'
                   else ['-m','zkinfer.runtime.node.supervisor'])
        command += [f'{k}={json.dumps(v)}' for k,v in values.items()]
        write(out/(role+'-command.json'), command)
        unit = unit_name(c, attempt, role)
        launch = ['sudo','systemd-run','--unit',unit,'--uid=ubuntu', '--collect',
                  '--working-directory='+str(ROOT), '--property=KillMode=control-group',
                  '--property=TimeoutStopSec=45', '--property=RuntimeMaxSec='+str(c['timeout_seconds']+1800),
                  '--property=StandardOutput=append:'+str(out/(role+'.log')),
                  '--property=StandardError=append:'+str(out/(role+'.log'))]
        for key in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','AWS_DEFAULT_REGION',
                    'CONDA_PREFIX','LD_LIBRARY_PATH'):
            if key not in os.environ: continue
            launch += ['--setenv='+key+'='+os.environ[key]]
        launch += ['--setenv=PYTHONPATH='+str(ROOT), '--setenv=PATH='+str(Path(sys.executable).parent)+':/usr/local/bin:/usr/bin:/bin',
                   '--setenv=ZKINFER_READY_FILE='+str(out/'registered.json'), *command]
        subprocess.run(launch, check=True)
        if role == 'coordinator':
            wait_for_coordinator(unit,c['port'],out/'coordinator.log')
        return
    if action == 'traffic':
        until = time.monotonic()+180
        expected = set(payload['instance_ids'])
        while time.monotonic()<until:
            try:
                from datetime import datetime, timezone
                state = json.loads((out/'registered.json').read_text())
                now = datetime.now(timezone.utc)
                live = {k for k,v in state.items() if (now-datetime.fromisoformat(v['timestamp'])).total_seconds()<30}
                if live == expected: break
            except (OSError, ValueError): pass
            time.sleep(2)
        else: raise TimeoutError('Expected worker registrations not received; inspect logs/network')
        command = [sys.executable,'evaluation/traffic.py','--target','127.0.0.1:'+str(c['port']),
                   '--workload',payload['workload'],'--requests',str(payload['requests']),
                   '--partition-size','1','--seed',str(4100+payload['trial']),
                   '--timeout',str(c['timeout_seconds']), '--output',str(out/'traffic')]
        from evaluation.campaign import launch
        launch(command, timeout=c['timeout_seconds']+30)
        summary = json.loads((out/'traffic/summary.json').read_text())
        n = payload['requests']
        if summary['request_count'] != n or summary['statuses'] != {'completed':n}:
            raise RuntimeError('Incomplete request batch; preserve failed attempt')
        write(out/'SUCCESS.json', summary)
        return
    if action == 'collect':
        for role in ('node','coordinator'):
            unit = unit_name(c,attempt,role)
            state = subprocess.run(['sudo','systemctl','show','-p','LoadState','--value',unit],
                                   text=True,capture_output=True)
            if state.stdout.strip() != 'not-found':
                subprocess.run(['sudo','systemctl','stop',unit],check=True)
        # Upload reports/logs/profile state, not large cache/tmp copies. Proofs already reside in S3.
        for p in out.rglob('*'):
            rel = p.relative_to(out)
            if p.is_file() and not any(part in {'cache','tmp'} for part in rel.parts):
                s3.upload_file(str(p), bucket, trial_prefix+'/nodes/'+payload['instance_id']+'/'+rel.as_posix())
        # Only disposable cache created by this exact attempt is removed, after upload.
        shutil.rmtree(out/'cache', ignore_errors=True)
        return
    raise ValueError(action)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('SSM command interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    main(json.loads(base64.b64decode(sys.argv[1])))
