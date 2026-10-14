"""CPU-only campaign tests; no backend proof generation or AWS calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parent/filename)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

c=load('campaign_test','campaign.py')
r=load('runner_test','run.py')

class CampaignTests(unittest.TestCase):
    def test_bootstrap_requires_explicit_sequential_preparation(self):
        base=['--baseline','fixed-1','--setup','warm','--setup-dir','/tmp/seed',
              '--bootstrap-setup','--allow-unprofiled']
        r.validate(r.parser().parse_args(base))
        for baseline in ['stage','whole-job','fixed-8']:
            with self.assertRaisesRegex(ValueError,'sequential'):
                r.validate(r.parser().parse_args([*base,'--baseline',baseline]))
        with self.assertRaises(ValueError):
            r.validate(r.parser().parse_args(base[:-1]))

    def test_resume_retains_failure_and_skips_only_success(self):
        with tempfile.TemporaryDirectory() as d:
            args=['prepare','--id','test-g1-cold','--root',d,'--workload','test',
                  '--condition','cold','--reserve','8','--execute']
            with patch.object(c,'prepare',side_effect=RuntimeError('failed proof')):
                with self.assertRaises(RuntimeError): c.main(args)
            parent=Path(d)/'profiles/test-g1-cold'
            self.assertTrue((parent/'attempt-001/FAILED.json').is_file())
            self.assertFalse((parent/'DONE.json').exists())
            with patch.object(c,'prepare') as prep:
                c.main(args); c.main(args)
                self.assertEqual(prep.call_count,1)
            self.assertTrue((parent/'attempt-002').is_dir())
            with self.assertRaisesRegex(ValueError,'Configuration changed'):
                c.main([*args,'--sampling','0.2'])

    def test_warm_history_excludes_bootstrap_and_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); task=root/'task'; task.mkdir()
            a=c.parser().parse_args(['prepare','--id','test-g1-warm','--root',d,
                '--workload','test','--reserve','8','--repetitions','3'])
            calls=[]
            def fake(command):
                args=r.parser().parse_args(list(map(str,command))[2:])
                calls.append(args)
                out=args.output; out.mkdir(parents=True)
                obs=[]
                if args.profile_dir:
                    obs=json.loads((args.profile_dir/'node.json').read_text())['stages']['proving']['observations']
                obs=[*obs,{'memory_gb':2}]
                p=out/'node-state/resource_profiles';p.mkdir(parents=True)
                (p/'node.json').write_text(json.dumps({'setup_key':'key','stages':{'proving':{'observations':obs}}}))
                p=out/'coordinator-profiles/key';p.mkdir(parents=True)
                (p/'resource_profile.json').write_text(json.dumps({'_resource_history':{'observations':[{'peak_memory_gb':2}]*len(obs)}}))
                p=out/'requests/request/reports';p.mkdir(parents=True)
                (p/'job_report.csv').write_text('setup_key,ezkl_key_gen_used_cache\nkey,True\n')
                (out/'run_status.json').write_text('{"status":"completed"}')
                (out/'traffic').mkdir()
                (out/'traffic/summary.json').write_text('{"request_count":1,"statuses":{"completed":1}}')
                if args.bootstrap_setup:
                    args.setup_dir.mkdir(); (args.setup_dir/'key').mkdir()
                    (args.setup_dir/'key/pk.key').write_text('test artifact')
            with patch.object(c,'launch',side_effect=fake): c.prepare(a,task)
            self.assertEqual(len(calls),4)
            self.assertTrue(calls[0].bootstrap_setup)
            self.assertIsNone(calls[1].profile_dir)
            self.assertEqual(calls[2].profile_dir,task/'observation-01/node-state/resource_profiles')
            final=json.loads((task/'snapshot/node/node.json').read_text())
            self.assertEqual(len(final['stages']['proving']['observations']),3)
            self.assertTrue(all(not x.bootstrap_setup for x in calls[1:]))

    def test_warm_audit_rejects_cache_miss(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'requests/r/reports').mkdir(parents=True)
            (p/'requests/r/reports/job_report.csv').write_text('setup_key,ezkl_key_gen_used_cache\nk,False\n')
            (p/'node-state/resource_profiles').mkdir(parents=True)
            (p/'node-state/resource_profiles/n.json').write_text(json.dumps({'setup_key':'k','stages':{'proving':{'observations':[{'memory_gb':1}]}}}))
            (p/'coordinator-profiles/k').mkdir(parents=True)
            (p/'coordinator-profiles/k/resource_profile.json').write_text(json.dumps({'_resource_history':{'observations':[{'peak_memory_gb':1}]}}))
            with self.assertRaisesRegex(ValueError,'cache hit'): c.audit_profiles(p,'warm',1)

    def test_failed_pilot_cannot_define_arrival_rate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'traffic').mkdir()
            (p/'run_status.json').write_text('{"status":"failed"}')
            (p/'traffic/summary.json').write_text('{"request_count":8,"statuses":{"completed":7,"failed":1},"drain_inclusive_successes_per_s":1}')
            with self.assertRaisesRegex(ValueError,'incomplete'): c.reference_rate(p)

    def test_monolithic_preparation_sends_valid_partition_argument(self):
        a=c.parser().parse_args(['prepare','--id','m','--root','/tmp/test','--workload','m','--partition','0','--reserve','8'])
        command=c.common(a,Path('/tmp/out'),'cold','monolithic')
        args=r.parser().parse_args(list(map(str,command))[2:])
        self.assertEqual(args.partition_size,1)
        self.assertIn('--monolithic',r.build_plan(args)['commands']['traffic'])

    def test_dry_campaign_does_not_create_state(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'absent'
            with contextlib.redirect_stdout(io.StringIO()):
                c.main(['prepare','--id','x','--root',str(p),'--workload','x','--reserve','8'])
            self.assertFalse(p.exists())

    def test_mixed_snapshots_do_not_copy_lock_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); task=root/'mixed'; task.mkdir()
            a=c.parser().parse_args(['prepare','--id','a,b-g1-warm','--root',d,
                                    '--workload','a,b','--reserve','8'])
            a.root=root
            for model in ['a','b']:
                source=root/'profiles'/f'{model}-g1-warm'; origin=source/'attempt-001'
                for rel in ['snapshot/node','snapshot/coordinator','setup']:
                    (origin/rel).mkdir(parents=True)
                    (origin/rel/f'{model}.json').write_text('{}')
                (origin/'snapshot/node/profiles.lock').write_text('')
                c.write_json(source/'DONE.json',{'attempt':'attempt-001','signature':{
                    'code':'hash','threads':{k:c.os.getenv(k) for k in c.THREADS}}})
            with patch.object(c,'code_fingerprint',return_value='hash'):
                c.prepare(a,task)
            self.assertEqual(sorted(p.name for p in (task/'snapshot/node').iterdir()),['a.json','b.json'])

if __name__=='__main__': unittest.main()
