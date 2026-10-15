# Per-run setup retention

`ZKINFER_KEEP_PROOF_ARTIFACTS=0` is already the runtime default. It suppresses
extra worker-report copies, not the VK/settings published for verification.
Do not delete those while a request is running or being finalized.

`run_all.sh` and `run_full.sh` now default to `PRUNE_RUN_SETUP=1`: after a
successful trial (including its configured verification), remove per-run PK,
VK and settings files. Set `PRUNE_RUN_SETUP=0` to preserve them for later
standalone verification. The separate existing `DISCARD_COPIES=1` setting
removes trial-setup and worker cache after successful trials.

This cleanup runs outside measured execution and does not change runtime code,
profile fingerprints, scheduling, cache initialization or proof verification.
Proofs, witnesses, reports, logs, traces, manifests, profiles and original
bootstrap setup are preserved. The original manifests can reference deleted
VK/settings: standalone re-verification requires restoring matching files.
Each removal is recorded in `setup_cleanup.jsonl`.

## Clean previous runs

Stop the campaign and its child runtime processes first. The tool holds the
same campaign lock and refuses execution while matching same-user runtime
processes from this checkout remain active. Do not launch other experiments
concurrently with cleanup. It is restricted to `ROOT/runs/*/attempt-*`, skips
symlinks, and does not touch `ROOT/bootstrap` or `ROOT/profiles`.

```bash
# Preview completed attempts only:
python evaluation/maintenance/prune_run_setup.py \
  --root evaluation/results/paper-day-v2 --discard-copies

# Delete the previewed files:
python evaluation/maintenance/prune_run_setup.py \
  --root evaluation/results/paper-day-v2 --discard-copies --execute

# Include interrupted/failed attempts ONLY after their processes have stopped:
python evaluation/maintenance/prune_run_setup.py \
  --root evaluation/results/paper-day-v2 --discard-copies \
  --include-incomplete --execute
```

`--discard-copies` removes entire attempt-local trial-setup and run/cache
folders as well as named setup files elsewhere in the attempt. This does not
remove the bootstrap setup. Without that option only PK/VK/settings files
are removed. Add `--run-id ID` to limit cleanup to one experiment.

The tool reports logical file sizes; actual disk recovery may differ for
shared/hard-linked/reflinked files. It does not clean profiling observations.
It also does not prevent transient disk use: warm runs still create trial-setup
and a fresh node cache. Do not restart a large trial with insufficient space
for those copies. Reducing peak disk demand requires a separate change.
