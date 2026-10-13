import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from experiments.microbench.repeatability import (
    select_jobs, settings_fingerprint, setup_key, fresh_output, percentile, write_json,
    prepare, worker,
)
from experiments.microbench.analysis import estimate, analyze_rows, measurements


def job(number, duration, memory):
    return {"job_status":"COMPLETED", "job_id":f"job_{number}",
            "job_name":f"model_sub_model_{number}", "setup_key":f"key_{number}",
            "job_runtime(s)":str(duration), "max_process_memory(GB)":str(memory)}


def row(rep, memory, condition="cold", fingerprint="same", stage="PROVING"):
    return {"model":"model", "partition":1, "condition":condition,
            "setup_key":"key", "settings_fingerprint":fingerprint, "stage":stage,
            "repetition":rep, "peak_memory_gib":memory, "duration_s":rep,
            "memory_samples":1 if memory is not None else 0,
            "avg_cpu_machine_pct":20}


class SelectionTests(unittest.TestCase):
    def test_interpolated_median_tie_uses_lower_partition_id(self):
        selected=select_jobs([job(4,12,1),job(2,4,2)])
        self.assertEqual(selected[0]["partition"],2)
        self.assertEqual(selected[0]["criteria"],["median_duration","maximum_memory"])

    def test_duration_and_memory_are_selected_independently(self):
        selected=select_jobs([job(1,1,50),job(2,10,1),job(3,100,2),job(4,200,3),job(5,300,4)])
        self.assertEqual([r["partition"] for r in selected],[3,5,1])

    def test_deduplicates_one_job_satisfying_all_roles(self):
        selected=select_jobs([job(7,10,20)])
        self.assertEqual(len(selected),1)
        self.assertEqual(len(selected[0]["criteria"]),3)

    def test_invalid_or_failed_populations_rejected(self):
        for rows in [[],[job(1,"nan",1)],[job(1,1,"inf")],[job(1,1,1),job(1,2,2)]]:
            with self.assertRaises(ValueError):select_jobs(rows)
        r=job(1,1,1);r["job_status"]="FAILED"
        with self.assertRaises(ValueError):select_jobs([r])

    def test_settings_fingerprint_ignores_json_order_but_detects_change(self):
        with TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            write_json(a,{"a":1,"b":2});write_json(b,{"b":2,"a":1})
            self.assertEqual(settings_fingerprint(a),settings_fingerprint(b))
            write_json(b,{"a":1,"b":3})
            self.assertNotEqual(settings_fingerprint(a),settings_fingerprint(b))

    def test_output_never_overwrites_existing_directory(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"bench";fresh_output(path)
            with self.assertRaises(FileExistsError):fresh_output(path)


class PredictionTests(unittest.TestCase):
    def test_estimator_warmup_and_switch(self):
        self.assertIsNone(estimate([10]))
        self.assertEqual(estimate([10,20]),20)
        self.assertEqual(estimate([10,20,30,40]),40)
        self.assertEqual(estimate([10,20,30,40,50]),46)

    def test_future_values_cannot_affect_previous_prediction(self):
        _,predictions,_=analyze_rows([row(1,1),row(2,2),row(3,1000),row(4,4)])
        memory=[r for r in predictions if r["metric"]=="peak_memory_gib"]
        self.assertEqual(memory[0]["predicted"],2)
        self.assertEqual(memory[0]["prior_observations"],2)
        self.assertEqual(memory[0]["shortfall"],998)
        self.assertEqual(memory[1]["predicted"],1000)

    def test_no_history_leak_between_conditions_or_settings(self):
        rows=[row(1,1),row(2,2),row(3,3),row(1,100,"warm"),row(2,200,"warm"),
              row(1,999,fingerprint="changed"),row(2,999,fingerprint="changed")]
        _,predictions,_=analyze_rows(rows)
        memory=[r for r in predictions if r["metric"]=="peak_memory_gib"]
        self.assertEqual(len(memory),1)
        self.assertEqual(memory[0]["condition"],"cold")

    def test_missing_memory_is_excluded_and_zero_is_observed(self):
        summaries,predictions,_=analyze_rows([row(1,0),row(2,None),row(3,2),row(4,3)])
        self.assertEqual(summaries[0]["memory_excluded_executions"],1)
        self.assertEqual(summaries[0]["memory_measured_executions"],3)
        memory=[r for r in predictions if r["metric"]=="peak_memory_gib"]
        self.assertEqual(len(memory),1)
        self.assertEqual(memory[0]["repetition"],4)
        self.assertEqual(memory[0]["prior_observations"],2)

    def test_repetition_order_and_duplicate_check(self):
        _,predictions,_=analyze_rows([row(3,3),row(1,1),row(2,2)])
        memory=[r for r in predictions if r["metric"]=="peak_memory_gib"]
        self.assertEqual(memory[0]["predicted"],2)
        with self.assertRaises(ValueError):analyze_rows([row(1,1),row(1,2)])


class TraceTests(unittest.TestCase):
    def make_trace(self, root, warmup=False):
        write_json(root/"execution.json",{"model":"model","partition":1,"condition":"cold",
                   "repetition":1,"warmup":warmup,"setup_key":"key","input_sha256":"input"})
        write_json(root/"result.json",{"ok":True,"settings_fingerprint":"settings",
                   "result":{"metrics":{}}})
        # Native events can be backfilled out of order. Wall-clock alignment
        # matters because logger and stage elapsed counters have different origins.
        (root/"stage_transitions.csv").write_text(
            "timestamp,elapsed_s,stage,event\n"
            "2026-01-01T00:00:00+00:00,0,CALIBRATING,enter\n"
            "2026-01-01T00:00:05+00:00,5,FUSED_KEYGEN_PROVE,enter\n"
            "2026-01-01T00:00:15+00:00,15,PUBLISHING_KEYS,enter\n"
            "2026-01-01T00:00:05.1+00:00,5.1,KEY_GEN,enter\n"
            "2026-01-01T00:00:10+00:00,10,PROVING,enter\n"
            "2026-01-01T00:00:14+00:00,14,PERSISTING_KEYS,enter\n"
            "2026-01-01T00:00:16+00:00,16,DONE,enter\n")
        (root/"resource_usage.csv").write_text(
            "timestamp,elapsed_s,type,stage,memory_gb,cpu_machine_percent\n"
            "2026-01-01T00:00:01+00:00,0,system,CALIBRATING,900,10\n"
            "2026-01-01T00:00:06+00:00,5,process_tree,KEY_GEN,5,20\n"
            "2026-01-01T00:00:11+00:00,10,process_tree,PROVING,8,30\n"
            "2026-01-01T00:00:14.5+00:00,13.5,process_tree,PERSISTING_KEYS,9,2\n")

    def test_wall_clock_alignment_unsampled_stage_and_fused_envelope(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_trace(root)
            rows={r["stage"]:r for r in measurements(root)}
            self.assertIsNone(rows["CALIBRATING"]["peak_memory_gib"])
            self.assertEqual(rows["PROVING"]["duration_s"],4)
            self.assertEqual(rows["PROVING"]["peak_memory_gib"],8)
            self.assertEqual(rows["FUSED_KEYGEN_PROVE_ENVELOPE"]["peak_memory_gib"],9)

    def test_warmup_is_not_a_measurement(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_trace(root,warmup=True)
            self.assertEqual(measurements(root),[])

    def test_cached_preparation_is_not_labelled_as_generation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_trace(root)
            result=json.loads((root/"result.json").read_text())
            result['result']['metrics']['ezkl_calibrate_settings_used_cache']=True
            write_json(root/"result.json",result)
            self.assertIn('SETTINGS_CACHE_LOOKUP',{r['stage'] for r in measurements(root)})


class PreparationTests(unittest.TestCase):
    def fixtures(self, root, changed=False, invariant=False):
        model=root/'model.onnx';model.write_bytes(b'model')
        baseline=root/'baseline.json';write_json(baseline,{'input_data':[[0.0]]})
        inputs=[]
        for k in range(1,4):
            path=root/f'input{k}.json';write_json(path,{'input_data':[[float(k)]]});inputs.append(str(path))
        config=root/'config.json'
        write_json(config,{'mnist_gan':{'model':str(model),'baseline_input':str(baseline),'inputs':inputs}})
        source=root/'source';source.mkdir()
        data=job(1,10,2);data['job_name']='mnist_gan_sub_model_1';data['setup_key']=setup_key(b'fixed component')
        with (source/'job_report.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data));w.writeheader();w.writerow(data)
        module=ModuleType('zkinfer.graph.onnx_splitter')
        def split(**kwargs):
            values=json.loads(Path(kwargs['input_data_path']).read_text())
            raw=b'changed' if changed and values['input_data'][0][0]>0 else b'fixed component'
            return [SimpleNamespace(name='sub_model_1',model=SimpleNamespace(SerializeToString=lambda:raw),
                                    input_data={'input_data':[[0.0]]} if invariant else values)]
        module.split_onnx_model_with_inputs=split
        args=SimpleNamespace(source=source,models=['mnist_gan'],config=config,output=root/'prepared',
                             repetitions=3,seed=0,input_mode='files',noise_scale=.01,integer_upper_bound=None)
        return args,module

    def test_full_model_inputs_materialize_fixed_partition(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);args,module=self.fixtures(root)
            with patch.dict('sys.modules',{'onnxruntime':ModuleType('onnxruntime'),
                                          'zkinfer.graph.onnx_splitter':module}):
                prepare(args)
            manifest=json.loads((args.output/'manifest.json').read_text())
            self.assertEqual(len(manifest['cases']),1)
            case=manifest['cases'][0]
            self.assertEqual(len(case['inputs']),4)
            self.assertEqual(json.loads((args.output/case['directory']/case['inputs'][3]).read_text()),
                             {'input_data':[[3.0]]})

    def test_changed_circuit_and_invariant_boundary_inputs_are_rejected(self):
        for changed,invariant in [(True,False),(False,True)]:
            with TemporaryDirectory() as tmp:
                args,module=self.fixtures(Path(tmp),changed,invariant)
                with patch.dict('sys.modules',{'onnxruntime':ModuleType('onnxruntime'),
                                              'zkinfer.graph.onnx_splitter':module}):
                    with self.assertRaises(ValueError):prepare(args)


class WorkerTests(unittest.TestCase):
    def fake_runtime(self, calls):
        module=ModuleType('zkinfer.runtime.node.prover_task')
        module.LocalProverTask=SimpleNamespace(from_dict=lambda data:SimpleNamespace(**data))
        class Stages:
            def __init__(self,**kwargs):
                self.input_data_path=kwargs['input_data_path']
                self.settings_path=str(Path(kwargs['proving_cache_root_dir'])/'settings.json')
            def calibrate_settings_with_fallback(self):
                calls.append(('calibration_input',self.input_data_path))
                hit=Path(self.settings_path).exists()
                write_json(self.settings_path,{'run_args':{'logrows':12}})
                return hit,0,0
        class Runner:
            def run(self,task):
                stages=self._build_proof_stages(task,str(Path(task.output_dir)/'status.txt'))
                hit,_,_=stages.calibrate_settings_with_fallback()
                calls.append(('proof_input',stages.input_data_path))
                metrics={k:hit for k in ['ezkl_calibrate_settings_used_cache',
                                        'ezkl_compile_circuit_used_cache','ezkl_key_gen_used_cache']}
                metrics['ezkl_fused_stage_trace_available']=True
                return SimpleNamespace(metrics=metrics,to_dict=lambda:{'metrics':metrics})
        module.ResourceAwareEZKLProofStages=Stages;module.ProverTaskRunner=Runner
        admission=ModuleType('zkinfer.runtime.node.phase_admission')
        admission.ResourceAdmissionController=lambda **kwargs:None
        return {'zkinfer.runtime.node.prover_task':module,
                'zkinfer.runtime.node.phase_admission':admission}

    def payload(self,root,condition='cold'):
        out=root/'run';out.mkdir()
        task={'output_dir':str(out),'input_path':'varied.json','setup_dir':str(root/'setup'),
              'model_path':'model.onnx','tmp_dir':str(root/'tmp'),'phase_admission_dir':str(root/'admission'),
              'setup_key':'key'}
        payload={'task':task,'condition':condition,'fingerprint_path':str(root/'fingerprint'),
                 'calibration_input':'reference.json','srs_dir':str(root/'srs'),'verify':False}
        path=root/'task.json';write_json(path,payload);return path,out

    def test_calibration_uses_reference_but_proof_input_is_restored(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,out=self.payload(root);calls=[]
            with patch.dict('sys.modules',self.fake_runtime(calls)):
                self.assertEqual(worker(path),0)
            self.assertEqual(calls,[('calibration_input','reference.json'),('proof_input','varied.json')])
            self.assertTrue(json.loads((out/'result.json').read_text())['ok'])
            self.assertTrue((out/'settings_snapshot.json').exists())

    def test_warm_cache_miss_fails_instead_of_becoming_cold_measurement(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,out=self.payload(root,condition='warm')
            with patch.dict('sys.modules',self.fake_runtime([])):
                self.assertEqual(worker(path),1)
            result=json.loads((out/'result.json').read_text())
            self.assertFalse(result['ok']);self.assertIn('cache flags',result['error'])

    def test_changed_settings_fingerprint_fails_before_proof(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,out=self.payload(root);(root/'fingerprint').write_text('different')
            calls=[]
            with patch.dict('sys.modules',self.fake_runtime(calls)):
                self.assertEqual(worker(path),1)
            self.assertEqual(calls,[('calibration_input','reference.json')])


if __name__=="__main__":unittest.main()
