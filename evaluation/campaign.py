#!/usr/bin/env python3
"""Preparation, isolated trial state and resumable checkpoints for run_all.sh."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
THREADS = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')

def read_json(path):
    return json.loads(path.read_text())

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)

def code_fingerprint():
    # Include local edits as well as committed source, excluding outputs/models.
    h = hashlib.sha256()
    paths = [*ROOT.joinpath('zkinfer').rglob('*.py'),
             *ROOT.joinpath('zkinfer/config').rglob('*.yaml'),
             *ROOT.joinpath('evaluation').glob('*.py'),
             ROOT/'evaluation/configs/baselines.json',
             *ROOT.joinpath('experiments/config/workload').glob('*.yaml')]
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

def copy_tree(source, dest):
    """Independent writable copies; use CoW when available, never hard links."""
    if dest.exists(): raise ValueError(f'Refusing existing destination: {dest}')
    dest.mkdir(parents=True)
    subprocess.run(['cp', '-a', '--reflink=auto', str(source) + '/.', str(dest)], check=True)

def launch(command, timeout=None):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    process = subprocess.Popen(list(map(str, command)), cwd=ROOT, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code: raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise

def succeeded(output):
    status = read_json(output/'run_status.json')
    summary = read_json(output/'traffic/summary.json')
    n = summary['request_count']
    return status['status'] == 'completed' and n > 0 and summary['statuses'] == {'completed': n}

def audit_profiles(output, condition, minimum):
    """Check every measured setup key, rather than merely nonempty directories."""
    reports = list(output.glob('requests/**/reports/job_report.csv'))
    rows = []
    for p in reports:
        with p.open(newline='') as f: rows.extend(csv.DictReader(f))
    if not rows: raise ValueError(f'No job reports in {output}')
    keys = {r['setup_key'] for r in rows}
    node = {}
    for p in (output/'node-state/resource_profiles').glob('*.json'):
        data = read_json(p); node[data['setup_key']] = data
    filename = 'cold_resource_profile.json' if condition == 'cold' else 'resource_profile.json'
    for key in keys:
        cp = read_json(output/'coordinator-profiles'/key/filename)
        observations = cp.get('_resource_history', {}).get('observations', [])
        if len(observations) < minimum or any(float(o.get('peak_memory_gb', 0)) <= 0 for o in observations):
            raise ValueError(f'Missing/invalid coordinator observations for {key}')
        stages = node.get(key, {}).get('stages', {})
        usable = [v.get('observations', []) for k,v in stages.items()
                  if not k.startswith('waiting') and 'prov' in k]
        if not any(len(obs) >= minimum and all(float(o.get('memory_gb',0)) > 0 for o in obs) for obs in usable):
            raise ValueError(f'Missing sampled proving-stage history for {key}; inspect sampling/backend instrumentation')
    if condition == 'warm':
        for r in rows:
            if str(r.get('ezkl_key_gen_used_cache', '')).lower() not in {'true','1','1.0'}:
                raise ValueError('Warm profiling did not report a proving-key cache hit for every job; inspect setup identity and cache capacity')
    return len(keys)

def reference_rate(path):
    """A finite-batch pilot reference, not a steady-state capacity estimate."""
    summary = read_json(path/'traffic/summary.json')
    if not succeeded(path): raise ValueError('Arrival-rate pilot was incomplete')
    rate = summary['drain_inclusive_successes_per_s']
    if rate is None or rate <= 0: raise ValueError('Invalid pilot completion rate')
    return rate

def common(a, output, condition, baseline, requests=1, seed=42, reserve=None):
    return [sys.executable, ROOT/'evaluation/run.py', '--execute', '--baseline', baseline,
            '--workload', a.workload, '--partition-size', str(max(1,a.partition)),
            '--requests', str(requests), '--seed', str(seed), '--setup', condition,
            '--reserve-gib', str(a.reserve if reserve is None else reserve),
            '--cache-gib', str(a.cache_gib), '--sampling-seconds', str(a.sampling),
            '--timeout', str(a.timeout), '--port', str(a.port), '--verify', '--output', output]

def prepare(a, task):
    """Share a setup-only bootstrap, then collect condition-matched proof history."""
    if ',' in a.workload:
        for model in a.workload.split(','):
            source = a.root/'profiles'/f'{model}-g{a.partition}-{a.condition}'
            done = read_json(source/'DONE.json')
            if done['signature']['code'] != code_fingerprint() or done['signature']['threads'] != {k:os.getenv(k) for k in THREADS}:
                raise ValueError('Mixed-workload snapshots must use the same code and thread settings')
            origin = source/done['attempt']
            for rel in ['snapshot/node','snapshot/coordinator'] + (['setup'] if a.condition == 'warm' else []):
                dest = task/rel; dest.mkdir(parents=True, exist_ok=True)
                source_dir = setup_source(origin) if rel == 'setup' else origin/rel
                for entry in source_dir.iterdir():
                    if entry.name == 'metadata.json': continue
                    if rel == 'snapshot/node' and entry.suffix != '.json': continue
                    target = dest/entry.name
                    if target.exists(): raise ValueError(f'Conflicting mixed-workload snapshot entry: {target}')
                    if entry.is_dir(): copy_tree(entry,target)
                    else: shutil.copy2(entry,target)
        write_json(task/'snapshot/metadata.json', {'workload':a.workload,'condition':a.condition,'partition':a.partition})
        return
    seed_setup = ensure_bootstrap(a)
    write_json(task/'setup_source.json', {'path':str(seed_setup)})
    baseline = 'monolithic' if a.partition == 0 else 'fixed-1'
    last = None
    # Setup-only bootstrap supplies SRS and reusable artifacts but no observations.
    # Cold runs do not receive its setup directory and still regenerate setup.
    for rep in range(1, a.repetitions+1):
        out = task/f'observation-{rep:02d}'
        cmd = common(a, out, a.condition, baseline) + ['--allow-unprofiled']
        if a.condition == 'warm': cmd += ['--setup-dir', seed_setup]
        if last:
            cmd += ['--profile-dir', last/'node-state/resource_profiles',
                    '--coordinator-profile-dir', last/'coordinator-profiles']
        launch(cmd)
        if not succeeded(out): raise ValueError('Isolated profiling request did not complete')
        audit_profiles(out, a.condition, rep)
        if a.discard_copies and (out/'cache').exists(): shutil.rmtree(out/'cache')
        last = out
    copy_tree(last/'node-state/resource_profiles', task/'snapshot/node')
    copy_tree(last/'coordinator-profiles', task/'snapshot/coordinator')
    write_json(task/'snapshot/metadata.json', dict(condition=a.condition, workload=a.workload,
               partition=a.partition, observations=a.repetitions, keys=audit_profiles(last,a.condition,a.repetitions)))

def profile_root(a):
    return a.root/'profiles'/f'{a.workload}-g{a.partition}-{a.condition}'

def setup_source(task):
    ref=task/'setup_source.json'
    return Path(read_json(ref)['path']) if ref.is_file() else task/'setup'

def ensure_bootstrap(a):
    parent=a.root/'bootstrap'/f'{a.workload}-g{a.partition}'
    signature={'workload':a.workload,'partition':a.partition,'code':code_fingerprint(),
               'threads':{k:os.getenv(k) for k in THREADS}}
    parent.mkdir(parents=True,exist_ok=True)
    record=parent/'bootstrap_plan.json'
    if record.exists() and read_json(record)!=signature:
        raise ValueError('Bootstrap configuration changed; use a new campaign root')
    write_json(record,signature)
    output=parent/'artifacts'
    if not (parent/'DONE.json').is_file():
        launch([sys.executable,ROOT/'evaluation/bootstrap_setup.py','--workload',a.workload,
                '--partition',str(a.partition),'--output',output],timeout=a.timeout)
        manifest=read_json(output/'manifest.json')
        if not manifest.get('setup_only') or manifest.get('proofs_generated')!=0 or not manifest.get('jobs'):
            raise ValueError('Invalid setup-only bootstrap manifest')
        write_json(parent/'DONE.json',signature)
    # Verify all bundles, including on resume; do not silently accept deleted keys.
    manifest=read_json(output/'manifest.json')
    for job in manifest['jobs']:
        for name in ('model.onnx','settings.json','network.compiled','pk.json','vk.json'):
            path=output/'setup'/job['setup_key']/name
            if not path.is_file() or path.stat().st_size==0:
                raise ValueError(f'Missing bootstrap artifact: {path}')
    return output/'setup'

def execute_cell(a, task):
    profiles = profile_root(a)
    if not (profiles/'DONE.json').is_file():
        raise ValueError(f'Run preparation first: {profiles}')
    prepared = read_json(profiles/'DONE.json')
    if prepared['signature']['code'] != code_fingerprint():
        raise ValueError('Code changed since profiling. Use a new campaign root and recollect profiles.')
    if prepared['signature']['threads'] != {k:os.getenv(k) for k in THREADS}:
        raise ValueError('Thread settings differ from profile collection')
    if prepared['signature']['arguments']['sampling'] != a.sampling:
        raise ValueError('Sampling differs from profile collection')
    data = profiles/prepared['attempt']
    out = task/'run'
    cmd = common(a, out, a.condition, a.baseline, a.requests, a.seed)
    cmd += ['--profile-dir', data/'snapshot/node', '--coordinator-profile-dir', data/'snapshot/coordinator']
    if a.condition == 'warm':
        # Reuse the immutable bootstrap setup directly.  The worker still restores
        # it into this trial's fresh node cache, so local cache state remains
        # isolated without physically duplicating the shared proving keys first.
        # Completed warm setup bundles are never republished by NodeSupervisor;
        # any per-run cache/profile/output state stays under this attempt's run/.
        seed_setup = setup_source(data).resolve()
        if not seed_setup.is_dir() or not any(seed_setup.iterdir()):
            raise ValueError(f'Warm setup source is missing or empty: {seed_setup}')
        write_json(task/'setup_source.json', {
            'path': str(seed_setup),
            'shared': True,
            'purpose': 'read-only bootstrap setup reused by measured warm trial',
        })
        cmd += ['--setup-dir', seed_setup]
    if a.arrival == 'poisson':
        pilot = a.root/'runs'/a.pilot
        marker = read_json(pilot/'DONE.json')
        pilot_out = pilot/marker['attempt']/'run'
        plan = read_json(pilot_out/'plan.json')['arguments']
        if plan['workload'] != a.workload or plan['setup'] != a.condition or plan['baseline'] != 'whole-job':
            raise ValueError('Pilot must use whole-job admission and the same workload/setup')
        if abs(plan['reserve_gib']-a.reserve) > .001:
            raise ValueError('Pilot budget differs from measured trial')
        rate = reference_rate(pilot_out)*a.rate_factor
        cmd += ['--arrival','poisson','--rate',str(rate)]
    launch(cmd)
    if not succeeded(out): raise ValueError('Trial incomplete; retained for failure analysis')
    if a.discard_copies:
        # The bootstrap setup is shared and must never be deleted here.  Only
        # disposable per-attempt node cache state is removed after success.
        if (out/'cache').exists(): shutil.rmtree(out/'cache')

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run'])
    p.add_argument('--id', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workload', required=True)
    p.add_argument('--partition', type=int, default=1, help='0 means monolithic for preparation')
    p.add_argument('--condition', choices=['cold','warm'], default='warm')
    p.add_argument('--baseline', default='stage')
    p.add_argument('--requests', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--reserve', type=float, required=True)
    p.add_argument('--cache-gib', type=float, default=500)
    p.add_argument('--sampling', type=float, default=.1)
    p.add_argument('--timeout', type=float, default=172800)
    p.add_argument('--port', type=int, default=50051)
    p.add_argument('--arrival', choices=['burst','poisson'], default='burst')
    p.add_argument('--pilot')
    p.add_argument('--rate-factor', type=float, default=1)
    p.add_argument('--discard-copies', action='store_true')
    p.add_argument('--execute', action='store_true')
    return p

def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    a = parser().parse_args(argv)
    if a.partition < 0 or a.repetitions < 1 or a.reserve < 0:
        raise ValueError('Invalid partition, repetitions or reserve')
    if Path(a.id).name != a.id or not a.id or a.id in {'.','..'}: raise ValueError('Invalid ID')
    if a.arrival == 'poisson' and (not a.pilot or a.rate_factor <= 0): raise ValueError('Poisson requires pilot and positive rate-factor')
    a.root = a.root.resolve()
    if not a.execute:
        baseline = ('monolithic' if a.partition == 0 else 'fixed-1') if a.action == 'prepare' else a.baseline
        print(f'PLAN {a.action} {a.id}: {a.workload} g={a.partition} {a.condition}; baseline={baseline}, N={a.requests}, reserve={a.reserve:g} GiB')
        return 0
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root/'campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = a.root/('profiles' if a.action == 'prepare' else 'runs')/a.id
        parent.mkdir(parents=True, exist_ok=True)
        signature = {'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                     'code': code_fingerprint(), 'threads': {k:os.getenv(k) for k in THREADS}}
        if (parent/'DONE.json').exists():
            old=read_json(parent/'DONE.json')
            if old['signature'] != signature:
                raise ValueError(f'Configuration changed for completed ID {a.id}; use a new campaign root/ID')
            print(f'SKIP completed {a.id}', flush=True); return 0
        # A failed/interrupted run is never overwritten or silently skipped.
        number = 1
        while (parent/f'attempt-{number:03d}').exists(): number += 1
        task = parent/f'attempt-{number:03d}'
        task.mkdir()
        write_json(task/'campaign_plan.json', signature)
        try:
            (prepare if a.action == 'prepare' else execute_cell)(a, task)
        except BaseException as exc:
            write_json(task/'FAILED.json', {'error':str(exc)})
            raise
        write_json(parent/'DONE.json', {'signature':signature,'attempt':task.name})
        print(f'DONE {a.id}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
