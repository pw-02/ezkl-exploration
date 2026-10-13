import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHIM = '''#!/usr/bin/env python3
import json, os, sys, subprocess
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    raise SystemExit(subprocess.call([sys.executable] + args))
if args[0] == '-':
    sys.argv = args
    exec(compile(sys.stdin.read(), '<stdin>', 'exec'))
    raise SystemExit(0)
with open(os.environ['FAKE_LOG'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
if args[1] == 'experiments.microbench.repeatability' and args[2] == 'prepare':
    if os.environ.get('FAKE_FAIL_PREPARE'):
        raise SystemExit(7)
    out = Path(args[args.index('--output') + 1]); out.mkdir(parents=True)
    models = args[args.index('--models')+1:args.index('--input-mode')]
    cases = []
    for model in models:
        for number, role in [(2,'median_duration'),(10,'p90_duration'),(7,'maximum_memory')]:
            cases.append({'model':model,'partition':number,'criteria':[role]})
    (out/'manifest.json').write_text(json.dumps({'cases':cases}))
elif args[1] == 'experiments.microbench.repeatability':
    Path(args[args.index('--output') + 1]).mkdir(parents=True)
else:
    (Path(args[args.index('--runs')+1])/'analysis').mkdir()
'''


class LauncherTests(unittest.TestCase):
    def run_script(self, name, extra=(), fail=False):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'source archive.tar.gz';source.write_text('stub')
            interpreter=root/'python shim';interpreter.write_text(SHIM);interpreter.chmod(0o755)
            log=root/'commands.jsonl'
            env={**os.environ,'PYTHON_BIN':str(interpreter),'FAKE_LOG':str(log)}
            if fail:env['FAKE_FAIL_PREPARE']='1'
            result=subprocess.run(['bash',str(ROOT/'experiments/microbench'/name),
                '--source',str(source),'--output-root',str(root/'outputs with spaces'),
                *extra],env=env,cwd=root,text=True,capture_output=True)
            commands=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result,commands

    def test_pilot_automatically_runs_only_median_selection(self):
        result,commands=self.run_script('run_pilot.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([c[2] if c[1].endswith('repeatability') else 'analysis' for c in commands],
                         ['prepare','run','analysis'])
        self.assertEqual(commands[0][commands[0].index('--repetitions')+1],'3')
        self.assertEqual(commands[1][commands[1].index('--case')+1],'mnist_gan:2')
        self.assertIn('--verify',commands[1])

    def test_main_defaults_to_all_models_without_case_filter(self):
        result,commands=self.run_script('run_main.sh')
        self.assertEqual(result.returncode,0,result.stderr)
        prepare=commands[0]
        self.assertEqual(prepare[prepare.index('--models')+1:prepare.index('--input-mode')],
                         ['mnist_gan','mobilenet_v2','nano_gpt_4_layers_64_embd'])
        self.assertEqual(prepare[prepare.index('--repetitions')+1],'10')
        self.assertNotIn('--case',commands[1])

    def test_model_alias_seed_and_integer_bound_are_forwarded(self):
        result,commands=self.run_script('run_main.sh',['--model','nanogpt','--seed','123',
            '--repetitions','12','--integer-upper-bound','50000'])
        self.assertEqual(result.returncode,0,result.stderr)
        prepare=commands[0]
        self.assertEqual(prepare[prepare.index('--models')+1],'nano_gpt_4_layers_64_embd')
        self.assertEqual(prepare[prepare.index('--seed')+1],'123')
        self.assertEqual(prepare[prepare.index('--integer-upper-bound')+1],'50000')

    def test_failed_prepare_does_not_run_proofs_or_analysis(self):
        result,commands=self.run_script('run_pilot.sh',fail=True)
        self.assertEqual(result.returncode,7)
        self.assertEqual(len(commands),1)
        self.assertIn('Campaign stopped',result.stderr)


if __name__=='__main__':unittest.main()
