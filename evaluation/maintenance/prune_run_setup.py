#!/usr/bin/env python3
"""Remove per-run setup copies after verification; never touch bootstrap/profiles.

Default is preview and completed attempts only. Stop the campaign before using
--include-incomplete. Deleting VK/settings prevents standalone re-verification
from the original per-run manifest; reports and proofs themselves are retained.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

NAMES = {'pk.json', 'pk.json.gz', 'vk.json', 'settings.json'}
PROTECTED = {'bootstrap', 'profiles', 'snapshot', 'node-state', 'coordinator-profiles'}


def setup_file(name):
    return name in NAMES or any(name.endswith('--'+n) for n in NAMES)


def files_without_links(path):
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in PROTECTED and not (Path(base)/d).is_symlink()]
        for name in files:
            p = Path(base)/name
            if not p.is_symlink() and p.is_file(): yield p


def attempts(root, run_id=None, include_incomplete=False):
    runs = root/'runs'
    if runs.is_symlink() or not runs.is_dir(): raise ValueError('Expected a real campaign runs directory')
    cells = [runs/run_id] if run_id else sorted(runs.iterdir())
    for cell in cells:
        if cell.is_symlink() or not cell.is_dir(): continue
        completed = None
        marker = cell/'DONE.json'
        if marker.is_file() and not marker.is_symlink():
            completed = json.loads(marker.read_text()).get('attempt')
        for attempt in sorted(cell.glob('attempt-*')):
            if attempt.is_symlink() or not attempt.is_dir(): continue
            if include_incomplete or attempt.name == completed: yield attempt


def candidates(attempt, discard_copies=False):
    trees = []
    if discard_copies:
        for tree in (attempt/'trial-setup', attempt/'run/cache'):
            if tree.is_dir() and not tree.is_symlink(): trees.append(tree)
    for tree in trees:
        # Refuse recursive removal of a directory containing links.
        for base, dirs, files in os.walk(tree, followlinks=False):
            if any((Path(base)/n).is_symlink() for n in dirs+files):
                raise ValueError('Refusing linked cache tree: '+str(tree))
        yield tree, sum(p.stat().st_size for p in files_without_links(tree)), True
    for p in files_without_links(attempt):
        if any(tree in p.parents for tree in trees): continue
        if setup_file(p.name): yield p, p.stat().st_size, False


def active_runtime_processes(repo):
    matches=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
        try:
            if entry.stat().st_uid != os.getuid(): continue
            cmd=(entry/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            relevant = ('zkinfer.runtime' in cmd or 'evaluation/campaign.py' in cmd or
                        'evaluation/run.py' in cmd or 'evaluation/bootstrap_setup.py' in cmd)
            if relevant:
                cwd=(entry/'cwd').resolve()
                if cwd == repo or repo in cwd.parents: matches.append((entry.name,cmd[:200]))
        except (FileNotFoundError, ProcessLookupError): continue
        except PermissionError:
            raise RuntimeError('Cannot inspect same-user process; cannot establish idle runtime')
    return matches


def prune(attempt, discard_copies, execute):
    chosen=list(candidates(attempt,discard_copies))
    total=sum(size for _,size,_ in chosen)
    print(f'{attempt}: {len(chosen)} targets, {total/1024**3:.3f} GiB logical size',flush=True)
    if execute and chosen:
        record=attempt/'setup_cleanup.jsonl'
        with record.open('a') as log:
            for p,size,is_tree in chosen:
                # Record intent first, so interrupted cleanup is auditable.
                log.write(json.dumps(dict(time=time.time(),path=str(p.relative_to(attempt)),
                    bytes=size,status='removing',reverification_artifacts_pruned=True))+'\n')
                log.flush()
                if is_tree: shutil.rmtree(p)
                else: p.unlink()
                log.write(json.dumps(dict(path=str(p.relative_to(attempt)),status='removed'))+'\n')
                log.flush()
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id')
    p.add_argument('--include-incomplete',action='store_true',help='Only after stopping all experiment processes')
    p.add_argument('--discard-copies',action='store_true',help='Also remove trial-setup and run/cache')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.run_id and (Path(a.run_id).name != a.run_id or a.run_id in {'.','..'}): p.error('Invalid run ID')
    root=a.root.resolve()
    if not (root/'runs').is_dir(): p.error('Campaign root must contain runs/')
    # Synchronize with campaign.py; do not race a new trial or profile collection.
    with (root/'campaign.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Campaign is active; stop it before pruning')
        if a.execute:
            active=active_runtime_processes(Path(__file__).resolve().parents[2])
            if active: raise SystemExit('Runtime processes still active: '+str(active))
        total=sum(prune(t,a.discard_copies,a.execute) for t in attempts(root,a.run_id,a.include_incomplete))
    print(('Removed' if a.execute else 'PREVIEW only: would remove'), f'{total/1024**3:.3f} GiB logical bytes')
    if not a.execute: print('Add --execute to delete. Bootstrap, profiles, reports, traces and proofs are preserved.')


if __name__=='__main__': main()
