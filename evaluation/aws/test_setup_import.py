import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evaluation.aws import setup_import as imp


class Missing(Exception):
    response = {'Error': {'Code': 'NoSuchKey'}}


class S3:
    def __init__(self, objects=None): self.objects = objects or {}; self.puts = []; self.fail = False
    def get_object(self, Bucket, Key):
        if Key not in self.objects: raise Missing()
        return {'Body': io.BytesIO(self.objects[Key])}
    def download_file(self, bucket, key, destination): Path(destination).write_bytes(self.objects[key])
    def head_object(self, **kwargs): raise Missing()
    def upload_file(self, path, bucket, key, **kwargs):
        if self.fail: raise RuntimeError('upload interrupted')
        self.objects[key] = Path(path).read_bytes()
    def put_object(self, Bucket, Key, Body): self.puts.append(Key); self.objects[Key] = Body


class Tests(unittest.TestCase):
    def bundle(self, root):
        key='a'*64
        sources={}
        for name in imp.REQUIRED:
            p=root/'source'/name; p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(json.dumps({'run_args':{'logrows':3}}).encode() if name=='settings.json' else b'artifact')
            sources[f'setup/{key}/{name}']=p
        p=root/'source/kzg3.srs';p.write_bytes(b'srs');sources['srs/kzg3.srs']=p
        m=dict(version=1,partition=1,workload='nano',jobs=[{'partition':0,'setup_key':key}],
               backend={'ezkl.so':'hash'},workload_identity={'onnx_file':'hash'},runtime_fingerprint='code',
               files={rel:dict(size=p.stat().st_size,sha256=imp.digest(p)) for rel,p in sources.items()})
        return m,sources

    def test_manifest_rejects_missing_key_and_path_escape(self):
        with tempfile.TemporaryDirectory() as t:
            m,_=self.bundle(Path(t));imp.validate_manifest(m)
            m['files']['../escape']=dict(size=1,sha256='b'*64)
            with self.assertRaises(ValueError): imp.validate_manifest(m)
            del m['files']['../escape']; del m['files']['setup/'+'a'*64+'/pk.json']
            with self.assertRaises(ValueError): imp.validate_manifest(m)

    def run_install(self, root, m, sources, **patches):
        s3=S3({'prefix/manifest.json':json.dumps(m).encode(),
               **{'prefix/'+rel:p.read_bytes() for rel,p in sources.items()}})
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name,value in [('backend_identity',m['backend']),('workload_identity',m['workload_identity']),('expected_jobs',m['jobs'])]:
                stack.enter_context(patch.object(imp,name,return_value=patches.get(name,value)))
            stack.enter_context(patch.object(imp,'ROOT',root/'repo'))
            stack.enter_context(patch('evaluation.campaign.code_fingerprint',return_value='code'))
            if patches.get('corrupt'): s3.objects['prefix/srs/kzg3.srs']=b'bad'
            imp.install(s3,'bucket','prefix','nano',root/'preparation')
        return s3

    def test_install_only_bootstrap_then_resume(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,sources=self.bundle(root)
            self.run_install(root,m,sources);self.run_install(root,m,sources)
            self.assertTrue((root/'preparation/bootstrap/nano-g1/DONE.json').is_file())
            self.assertFalse((root/'preparation/profiles').exists())
            self.assertEqual(json.loads((root/'preparation/bootstrap/nano-g1/artifacts/manifest.json').read_text())['proofs_generated'],0)

    def test_backend_mismatch_does_not_complete(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,sources=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'native EZKL'):
                self.run_install(root,m,sources,backend_identity={'different':'build'})
            self.assertFalse((root/'preparation').exists())

    def test_partition_mismatch_does_not_complete(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,sources=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'partition identities'):
                self.run_install(root,m,sources,expected_jobs=[])
            self.assertFalse((root/'preparation').exists())

    def test_bad_checksum_does_not_complete(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);m,sources=self.bundle(root)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                self.run_install(root,m,sources,corrupt=True)
            self.assertFalse((root/'preparation/bootstrap/nano-g1/DONE.json').exists())

    def test_upload_completion_written_last_and_no_aws_seed_done(self):
        with tempfile.TemporaryDirectory() as t:
            m,sources=self.bundle(Path(t));s3=S3()
            imp.publish(s3,'bucket','prefix',m,sources)
            self.assertEqual(s3.puts,['prefix/manifest.json'])
            self.assertEqual(len(s3.objects),len(sources)+1)

    def test_failed_upload_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as t:
            m,sources=self.bundle(Path(t));s3=S3();s3.fail=True
            with self.assertRaises(RuntimeError): imp.publish(s3,'bucket','prefix',m,sources)
            self.assertEqual(s3.puts,[])

    def test_existing_import_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as t:
            m,sources=self.bundle(Path(t));s3=S3({'prefix/manifest.json':b'{}'})
            with self.assertRaisesRegex(ValueError,'different artifacts'): imp.publish(s3,'bucket','prefix',m,sources)
            self.assertEqual(s3.puts,[])


if __name__=='__main__': unittest.main()
