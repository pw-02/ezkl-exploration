import json
from pathlib import Path
import tempfile
import unittest
from evaluation.maintenance.prune_run_setup import attempts, candidates, prune, setup_file


class Tests(unittest.TestCase):
    def test_exact_filenames(self):
        for n in ['pk.json','pk.json.gz','vk.json','lease--settings.json']:
            self.assertTrue(setup_file(n))
        for n in ['resource_profile.json','plan.json','settings.json.backup','proof.pf']:
            self.assertFalse(setup_file(n))

    def test_only_successful_attempt_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cell=root/'runs/test';cell.mkdir(parents=True)
            for n in ['attempt-001','attempt-002']: (cell/n).mkdir()
            self.assertEqual(list(attempts(root)),[])
            (cell/'DONE.json').write_text(json.dumps({'attempt':'attempt-002'}))
            self.assertEqual(list(attempts(root)),[cell/'attempt-002'])
            self.assertEqual(len(list(attempts(root,include_incomplete=True))),2)

    def test_preview_preserves_everything_and_execute_preserves_reports(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)
            paths=['run/proofs/x--vk.json','run/proofs/x--settings.json','run/proofs/proof.pf',
                   'run/reports/job_report.csv','run/node-state/resource_profiles/pk.json',
                   'run/snapshot/settings.json','run/profiles/vk.json','run/bootstrap/pk.json']
            for rel in paths:
                p=a/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('data')
            prune(a,False,False)
            self.assertTrue(all((a/rel).exists() for rel in paths))
            prune(a,False,True)
            self.assertFalse((a/paths[0]).exists());self.assertFalse((a/paths[1]).exists())
            self.assertTrue(all((a/rel).exists() for rel in paths[2:]))
            self.assertTrue((a/'setup_cleanup.jsonl').is_file())
            self.assertEqual(list(candidates(a)),[])

    def test_symlinks_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'attempt';a.mkdir();external=root/'bootstrap';external.mkdir()
            (external/'pk.json').write_text('key')
            (a/'linked').symlink_to(external,target_is_directory=True)
            (a/'vk.json').symlink_to(external/'pk.json')
            self.assertEqual(list(candidates(a)),[])
            self.assertTrue((external/'pk.json').exists())

    def test_discard_copies_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)
            for rel in ['trial-setup/k/model.onnx','run/cache/setup/k/network.compiled','run/reports/a.csv']:
                p=a/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('data')
            self.assertEqual(list(candidates(a)),[])
            prune(a,True,True)
            self.assertFalse((a/'trial-setup').exists());self.assertFalse((a/'run/cache').exists())
            self.assertTrue((a/'run/reports/a.csv').exists())

    def test_linked_cache_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d);(a/'trial-setup').mkdir();(a/'trial-setup/pk.json').symlink_to('/outside')
            with self.assertRaises(ValueError): list(candidates(a,True))


if __name__=='__main__': unittest.main()
