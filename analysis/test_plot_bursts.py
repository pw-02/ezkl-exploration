import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('plot_bursts', Path(__file__).with_name('plot_bursts.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    parent = root / 'runs/burst-bert-stage'
    run = parent / 'attempt-002/run'
    write(parent / 'DONE.json', {'attempt': 'attempt-002'})
    write(parent / 'attempt-001/run/plan.json', {})
    write(run / 'plan.json', {
        'arguments': {'baseline': 'stage', 'workload': 'bert', 'requests': 4,
                      'seed': 4101, 'setup': 'warm', 'reserve_gib': 627.5},
        'runtime_overrides': {'coordinator.scheduling_policy': 'resource_aware'},
    })
    write(run / 'run_status.json', {'status': 'completed'})
    write(run / 'traffic/summary.json', {
        'request_count': 4, 'statuses': {'completed': 4},
        'successful_request_latency_median_s': 120,
        'batch_makespan_s': 240,
    })
    return run


class PlotTests(unittest.TestCase):
    def test_done_selection_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = fixture(root)
            self.assertEqual(m.discover(root), [run])
            self.assertEqual(m.discover(run.parent.parent), [run])
            row = m.load_trial(run)
            self.assertEqual(row['workload'], 'TinyBERT')
            self.assertEqual(row['median_latency_min'], 2)
            self.assertEqual(row['makespan_min'], 4)
            m.validate_trials([row])

    def test_partial_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            run = fixture(Path(directory))
            write(run / 'traffic/summary.json', {'request_count': 4, 'statuses': {'completed': 3, 'failed': 1}})
            self.assertIsNone(m.load_trial(run))

    def test_duplicates_and_mixed_budgets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row = m.load_trial(fixture(Path(directory)))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                m.validate_trials([row, dict(row)])
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                m.validate_trials([row, dict(row, policy='Whole-job', reserve_gib=499.5)])

    def test_plot_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = m.load_trial(fixture(root))
            m.plot([row], ['TinyBERT'], [1, 4, 8, 16], root / 'figure')
            self.assertTrue((root / 'figure.pdf').is_file())
            self.assertTrue((root / 'figure.png').is_file())


if __name__ == '__main__':
    unittest.main()
