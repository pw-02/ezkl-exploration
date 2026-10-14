import importlib.util
import json
from pathlib import Path
import tempfile
import io
import zipfile
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('aws_control',HERE/'control.py')
control=importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
spec2=importlib.util.spec_from_file_location('aws_remote',HERE/'remote.py')
remote=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(remote)


class Tests(unittest.TestCase):
    def setUp(self): self.c=json.loads((HERE/'config.json').read_text())

    def test_default_matrix_has_matched_one_node_and_twenty_cells(self):
        control.validate(self.c)
        rows=[row for n in self.c['nodes'] for row in control.cells(self.c,n)]
        self.assertEqual(len(rows),20)
        self.assertEqual(len({k for k,v in rows}),20)
        self.assertEqual(self.c['nodes'][0],1)
        self.assertEqual({v['requests'] for k,v in rows},{1,16})

    def test_duplicate_or_descending_nodes_rejected(self):
        for nodes in ([1,2,2],[4,2],[0,1]):
            self.c['nodes']=nodes
            with self.assertRaises(ValueError): control.validate(self.c)

    def test_campaign_path_traversal_rejected(self):
        self.c['campaign']='../../old-results'
        with self.assertRaises(ValueError): control.validate(self.c)

    def test_launcher_repair_rejects_runtime_changes(self):
        def bundle(files):
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w') as z:
                for name,value in files.items(): z.writestr(name,value)
            return b.getvalue()
        old=bundle({'evaluation/aws/coordinator.py':'old','zkinfer/runtime/coordinator.py':'same'})
        new=bundle({'evaluation/aws/coordinator.py':'fixed','zkinfer/runtime/coordinator.py':'same'})
        self.assertEqual(control.launcher_only_changes(old,new),['evaluation/aws/coordinator.py'])
        bad=bundle({'evaluation/aws/coordinator.py':'fixed','zkinfer/runtime/coordinator.py':'changed'})
        with self.assertRaises(ValueError): control.launcher_only_changes(old,bad)

    def test_failed_coordinator_reports_log_before_workers_start(self):
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'coordinator.log';log.write_text('Primary config module not found')
            with patch.object(remote.subprocess,'run',return_value=Mock(stdout='failed\n')), patch.object(remote.socket,'create_connection') as connect:
                with self.assertRaisesRegex(RuntimeError,'Primary config module'):
                    remote.wait_for_coordinator('unit',50051,log)
                connect.assert_not_called()

    def test_wrapper_composes_directory_config_with_real_hydra(self):
        try: import hydra
        except ImportError: self.skipTest('hydra-core required for config regression test')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); wrapper=root/'evaluation/aws/coordinator.py';wrapper.parent.mkdir(parents=True)
            wrapper.write_text((HERE/'coordinator.py').read_text())
            config=root/'zkinfer/config';config.mkdir(parents=True)
            (config/'config.yaml').write_text('coordinator:\n  host: localhost\n  port: 50051\n')
            # Stub only the expensive server imports. Hydra itself composes a
            # real config directory without __init__.py and handles CLI overrides.
            code='''import sys,types,runpy
m=types.ModuleType('zkinfer.runtime.coordinator')
m.Coordinator=type('Coordinator',(),{'_record_node_locked':lambda *a:None})
sys.modules[m.__name__]=m
g=types.ModuleType('zkinfer.runtime.coordinator_grpc')
g.serve=types.SimpleNamespace(__wrapped__=lambda cfg: print('BOUND',cfg.coordinator.host,cfg.coordinator.port))
sys.modules[g.__name__]=g
sys.argv=[sys.argv[1],'coordinator.host=0.0.0.0','coordinator.port=50123']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            r=subprocess.run([sys.executable,'-c',code,str(wrapper)],cwd=root,text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            self.assertIn('BOUND 0.0.0.0 50123',r.stdout)

    def test_trial_order_rotates(self):
        self.c['trials']=2
        rows=list(control.cells(self.c,1))
        self.assertEqual(rows[0][1]['baseline'],'whole-job')
        self.assertEqual(rows[4][1]['baseline'],'stage')

    def test_no_execute_never_initializes_aws(self):
        with patch('sys.argv',['control.py','run']), patch.object(control,'Controller') as controller, patch('builtins.print'):
            control.main()
        controller.assert_not_called()

    def test_service_names_are_scoped_to_attempt(self):
        a=remote.unit_name(self.c,'trial/attempt-1','node')
        b=remote.unit_name(self.c,'trial/attempt-2','node')
        self.assertNotEqual(a,b)
        self.assertRegex(a,r'^zkinfer-eval-[a-f0-9]{16}-node$')

    def test_s3_download_cannot_escape_destination(self):
        s3=Mock()
        s3.get_paginator.return_value.paginate.return_value=[{'Contents':[{'Key':'prefix/../../escape'}]}]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): remote.download(s3,'bucket','prefix',Path(d))
        s3.download_file.assert_not_called()

    def test_existing_active_attempt_blocks_new_run(self):
        controller=object.__new__(control.Controller)
        with tempfile.TemporaryDirectory() as d:
            controller.local=Path(d)
            (Path(d)/'active.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'recover'): controller.run()

    def test_failed_ssm_command_is_not_success(self):
        controller=object.__new__(control.Controller)
        controller.ssm=Mock()
        controller.ssm.get_command_invocation.return_value={'Status':'Failed','ResponseCode':1}
        with self.assertRaises(RuntimeError): controller.wait('i-test','cmd',10)

    def test_shell_payload_does_not_interpolate_user_code(self):
        script=control.shell_command(self.c,'print("$(touch /tmp/not-run)")')
        self.assertNotIn('$(touch',script)
        self.assertIn('runuser -u ubuntu',script)

    def test_failed_trial_collects_all_nodes_without_completion_marker(self):
        controller=object.__new__(control.Controller)
        controller.c={**self.c,'nodes':[1],'requests':[1]}
        controller.prefix='evaluation/test'
        controller.s3=Mock()
        error=RuntimeError('missing'); error.response={'Error':{'Code':'404'}}
        controller.s3.head_object.side_effect=error
        controller.launch=Mock(return_value=[{'InstanceId':'i-one','PrivateIpAddress':'10.0.0.1'}])
        controller.install=Mock(); controller.freeze=Mock(); controller.stop=Mock()
        def call(iid,action,**kwargs):
            if action=='traffic': raise RuntimeError('proof failed')
        controller.call=Mock(side_effect=call)
        with tempfile.TemporaryDirectory() as d, patch.object(control,'source_bundle',return_value=(b'zip','hash')):
            controller.local=Path(d)
            with self.assertRaisesRegex(RuntimeError,'proof failed'): controller.run()
            self.assertTrue((Path(d)/'active.json').exists())
        self.assertEqual(controller.call.call_args_list[-1].args,('i-one','collect'))
        controller.s3.put_object.assert_not_called()
        controller.stop.assert_not_called()

    def test_success_marks_done_only_after_collection(self):
        controller=object.__new__(control.Controller)
        controller.c={**self.c,'nodes':[1],'requests':[1]}
        controller.prefix='evaluation/test'
        controller.s3=Mock()
        error=RuntimeError('missing'); error.response={'Error':{'Code':'404'}}
        controller.s3.head_object.side_effect=error
        controller.launch=Mock(return_value=[{'InstanceId':'i-one','PrivateIpAddress':'10.0.0.1'}])
        controller.install=Mock(); controller.freeze=Mock(); controller.stop=Mock()
        events=[]
        controller.call=Mock(side_effect=lambda iid,action,**kw:events.append(action))
        controller.s3.put_object.side_effect=lambda **kw:events.append('done')
        with tempfile.TemporaryDirectory() as d, patch.object(control,'source_bundle',return_value=(b'zip','hash')):
            controller.local=Path(d); controller.run()
            self.assertFalse((Path(d)/'active.json').exists())
        self.assertEqual(events.count('done'),2)
        for index,event in enumerate(events):
            if event=='done': self.assertEqual(events[index-1],'collect')
        controller.stop.assert_called_once()


if __name__=='__main__': unittest.main()
