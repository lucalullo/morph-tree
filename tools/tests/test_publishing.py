"""Offline adversarial publishing tests: no tokens, upload, model fit or network."""
import ast, copy, hashlib, importlib.util, io, json, os, socket, sys, tarfile, tempfile, unittest, urllib.error, zipfile
from pathlib import Path
from unittest.mock import patch
import yaml

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
gate=load('publication_gate_tests',ROOT/'tools/publish_reviewed.py')
producer=load('build_provenance_tests',ROOT/'tools/write_build_provenance.py')
COMMIT='a'*40

class Publishing(unittest.TestCase):
    def setUp(self):
        self.network=patch.object(socket.socket,'connect',side_effect=AssertionError('Tests cannot use network'));self.network.start()
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.run=dict(id=123,run_attempt=1,repository=dict(full_name=gate.REPOSITORY),head_repository=dict(full_name=gate.REPOSITORY),
            head_sha=COMMIT,head_branch='main',path='.github/workflows/test.yml',event='workflow_dispatch',status='completed',conclusion='success')
        self.jobs=[dict(name=n,status='completed',conclusion='success',head_sha=COMMIT) for n in sorted(gate.PROFILES)]
        self.artifacts=[dict(name=n,id=i+1,expired=False,size_in_bytes=100,workflow_run=dict(head_sha=COMMIT)) for i,n in enumerate(sorted(gate.ARTIFACTS))]
        self.environment=dict(name='testpypi',protection_rules=[dict(type='required_reviewers',reviewers=[dict(type='User')])])
    def tearDown(self):self.network.stop();self.temp.cleanup()
    def origin(self):return gate.validate_origin(self.run,self.jobs,self.artifacts,self.environment,COMMIT,123,1)
    def test_success_origin(self):self.assertTrue(self.origin()['required_reviewers_verified'])
    def test_reject_wrong_run_source_or_attempt(self):
        for field,value in [('id',124),('run_attempt',2),('head_sha','b'*40),('head_branch','feature'),('path','.github/workflows/other.yml'),('event','pull_request'),('status','in_progress'),('conclusion','failure')]:
            old=self.run;self.run=copy.deepcopy(old);self.run[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.origin()
            self.run=old
    def test_reject_fork_source(self):
        for key in ('repository','head_repository'):
            old=self.run[key];self.run[key]=dict(full_name='another/fork')
            with self.assertRaises(ValueError):self.origin()
            self.run[key]=old
    def test_reject_failed_skipped_missing_duplicate_jobs(self):
        for conclusion in ('skipped','cancelled','failure'):
            self.jobs[0]['conclusion']=conclusion
            with self.assertRaises(ValueError):self.origin()
        self.jobs[0]['conclusion']='success';self.jobs.pop()
        with self.assertRaises(ValueError):self.origin()
        self.jobs=[self.jobs[0]]*6
        with self.assertRaises(ValueError):self.origin()
    def test_reject_missing_expired_wrong_commit_artifacts(self):
        self.artifacts[0]['expired']=True
        with self.assertRaises(ValueError):self.origin()
        self.artifacts[0]['expired']=False;self.artifacts[0]['workflow_run']['head_sha']='b'*40
        with self.assertRaises(ValueError):self.origin()
        self.artifacts.pop()
        with self.assertRaises(ValueError):self.origin()
    def test_reject_missing_environment_reviewers(self):
        self.environment['protection_rules']=[]
        with self.assertRaises(ValueError):self.origin()
    def test_origin_api_responses_mocked(self):
        values=iter([self.run,dict(total_count=6,jobs=self.jobs),dict(total_count=6,artifacts=self.artifacts),self.environment]);seen=[]
        def fetch(url,token):seen.append(url);return next(values)
        self.assertEqual(gate.inspect_origin(COMMIT,'123',1,'fake-test-token',fetch)['commit'],COMMIT)
        self.assertEqual(len(seen),4)
        self.assertTrue(all(v.startswith('https://api.github.com/repos/lucalullo/morph-tree/') for v in seen))
    def test_reject_incomplete_api_pagination(self):
        values=iter([self.run,dict(total_count=7,jobs=self.jobs),dict(total_count=6,artifacts=self.artifacts)])
        with self.assertRaises(ValueError):gate.inspect_origin(COMMIT,'123',1,'unused',lambda *args:next(values))
    def test_index_404_only(self):
        def absent(url):raise urllib.error.HTTPError(url,404,'Not Found',{},None)
        self.assertTrue(gate.require_version_absent(absent)['absent'])
        with self.assertRaises(ValueError):gate.require_version_absent(lambda url:{'info':{'version':'2.1.0'}})
        for code in (401,403,429,500):
            def failed(url):raise urllib.error.HTTPError(url,code,'Unavailable',{},None)
            with self.assertRaises(urllib.error.HTTPError):gate.require_version_absent(failed)
        with self.assertRaises(TimeoutError):gate.require_version_absent(lambda url:(_ for _ in ()).throw(TimeoutError()))
    def test_no_redirect_or_token_forwarding(self):
        with self.assertRaises(ValueError):gate.NoRedirect().redirect_request(None,None,None,None,None,None)
        with self.assertRaises(ValueError):gate.get_json('https://test.pypi.org/pypi/morphtree/2.1.0/json','fake')
        with self.assertRaises(ValueError):gate.get_json('https://example.invalid/')
    def fixture(self):
        root=self.root/'project';(root/'tools').mkdir(parents=True);(root/'docs').mkdir()
        (root/'README.md').write_bytes(b'# MorphTree\n\nExact approved description.\n')
        (root/'pyproject.toml').write_text('[project]\nname="morphtree"\nversion="2.1.0"\nreadme={file="README.md",content-type="text/markdown"}\n',encoding='utf8')
        (root/'tools/check_distribution.py').write_bytes((ROOT/'tools/check_distribution.py').read_bytes())
        status=dict(version='2.1.0',version_approved=True,current_distribution_training_verified=True,local_technical_status='GO_SCOPED_LOCAL_DELIVERY',publication_approved=False,pypi_upload_enabled=False,testpypi_workflow=dict(prepared=True))
        (root/'docs/RELEASE_STATUS.json').write_text(json.dumps(status),encoding='utf8')
        source={p.relative_to(root).as_posix():gate.sha(p) for p in root.rglob('*') if p.is_file()}
        gate.publish(root/'SOURCE_MANIFEST.json',dict(files=source))
        return root
    def buildset(self,root,system):
        folder=self.root/system;folder.mkdir()
        tag='win_amd64' if system=='Windows' else 'manylinux_2_17_x86_64'
        wheel=folder/('morphtree-2.1.0-py3-none-'+tag+'.whl')
        metadata=b'Metadata-Version: 2.4\nName: morphtree\nVersion: 2.1.0\nDescription-Content-Type: text/markdown\n\n'+(root/'README.md').read_bytes()
        with zipfile.ZipFile(wheel,'w') as z:
            z.writestr('morphtree-2.1.0.dist-info/METADATA',metadata)
            z.writestr('morphtree/_experimental/manifest.json',json.dumps(dict(public_description=dict(README_sha256=gate.sha(root/'README.md')))))
        sdist=folder/'morphtree-2.1.0.tar.gz'
        with tarfile.open(sdist,'w:gz') as t:
            for p in sorted(root.rglob('*')):
                if p.is_file():
                    data=p.read_bytes();entry=tarfile.TarInfo('morphtree-2.1.0/'+p.relative_to(root).as_posix());entry.size=len(data);entry.mode=0o644 if system=='Windows' else 0o755;t.addfile(entry,io.BytesIO(data))
            entry=tarfile.TarInfo('morphtree-2.1.0/PKG-INFO');entry.size=len(metadata);t.addfile(entry,io.BytesIO(metadata))
        env=dict(GITHUB_SHA=COMMIT,GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_REF='refs/heads/main',GITHUB_WORKFLOW_REF=gate.REPOSITORY+'/.github/workflows/test.yml@refs/heads/main',GITHUB_REPOSITORY=gate.REPOSITORY,GITHUB_RUN_ID='123',GITHUB_RUN_ATTEMPT='1',RUNNER_OS=system)
        value=producer.manifest(env,[wheel,sdist],root,platform=system,python='3.12')
        gate.publish(folder/'BUILD_PROVENANCE.json',value)
        return folder,gate.sha(folder/'BUILD_PROVENANCE.json')
    def test_production_hard_block_even_with_global_go(self):
        root=self.fixture();gate.require_policy('testpypi','2.1.0',root)
        with self.assertRaises(ValueError):gate.require_policy('pypi','2.1.0',root)
        with self.assertRaises(ValueError):gate.require_policy('testpypi','3.0.0',root)
        self.assertFalse(gate.version_allowed('2.1.0','pypi'))
        with patch.object(sys,'argv',['publish_reviewed.py','--target','pypi']),self.assertRaisesRegex(SystemExit,'hard-blocked'):gate.main()
        with patch.object(sys,'argv',['publish_reviewed.py']),self.assertRaisesRegex(SystemExit,'verification-only'):gate.main()
    def test_valid_platform_sources_and_explicit_canonical_sdist(self):
        root=self.fixture();windows,wsha=self.buildset(root,'Windows');linux,lsha=self.buildset(root,'Linux')
        self.assertNotEqual(gate.sha(windows/'morphtree-2.1.0.tar.gz'),gate.sha(linux/'morphtree-2.1.0.tar.gz'))
        value=gate.prepare_bundle(windows,linux,wsha,lsha,self.origin(),self.root/'approved',root)
        self.assertEqual(len(value['files']),3)
        self.assertEqual(value['files']['morphtree-2.1.0.tar.gz'],gate.sha(windows/'morphtree-2.1.0.tar.gz'))
        with self.assertRaises(FileExistsError):gate.prepare_bundle(windows,linux,wsha,lsha,self.origin(),self.root/'approved',root)
    def test_reject_unapproved_manifest_artifact_and_extra_file(self):
        root=self.fixture();folder,pin=self.buildset(root,'Windows');origin=self.origin()
        with self.assertRaises(ValueError):gate.check_build_manifest(folder,'0'*64,origin,'Windows',root)
        wheel=next(folder.glob('*.whl'));raw=wheel.read_bytes();wheel.write_bytes(raw+b'changed')
        with self.assertRaises(ValueError):gate.check_build_manifest(folder,pin,origin,'Windows',root)
        wheel.write_bytes(raw);(folder/'extra.txt').write_text('x')
        with self.assertRaises(ValueError):gate.check_build_manifest(folder,pin,origin,'Windows',root)
    def test_reject_wrong_provenance_even_if_manifest_hash_reapproved(self):
        root=self.fixture();folder,pin=self.buildset(root,'Windows');path=folder/'BUILD_PROVENANCE.json';original=gate.read(path)
        for key,value in [('commit','b'*40),('build_run_attempt',2),('source_manifest_sha256','0'*64),('python','3.14'),('version','2.0.0')]:
            item=dict(original);item[key]=value;path.write_text(json.dumps(item))
            with self.subTest(key=key),self.assertRaises(ValueError):gate.check_build_manifest(folder,gate.sha(path),self.origin(),'Windows',root)
    def test_reject_source_tar_traversal_and_duplicate(self):
        for kind in ('traversal','duplicate'):
            path=self.root/(kind+'.tar.gz')
            with tarfile.open(path,'w:gz') as t:
                for name in (['../outside'] if kind=='traversal' else ['same','same']):
                    item=tarfile.TarInfo(name);item.size=1;t.addfile(item,io.BytesIO(b'x'))
            with self.assertRaises(ValueError):gate.sdist_payload(path)
    def test_reject_wheel_traversal_duplicate_and_links(self):
        for kind in ('traversal','duplicate','link'):
            path=self.root/(kind+'.whl')
            with zipfile.ZipFile(path,'w') as archive:
                if kind=='link':
                    entry=zipfile.ZipInfo('link');entry.create_system=3;entry.external_attr=0o120777<<16;archive.writestr(entry,'outside')
                else:
                    for name in (['../outside'] if kind=='traversal' else ['same','same']):archive.writestr(name,b'x')
            with self.assertRaises(ValueError):gate.safe_wheel(path)
    def test_reject_changed_source_payload(self):
        root=self.fixture();windows,wsha=self.buildset(root,'Windows');linux,lsha=self.buildset(root,'Linux')
        (root/'README.md').write_bytes(b'changed')
        with self.assertRaises(ValueError):gate.prepare_bundle(windows,linux,wsha,lsha,self.origin(),self.root/'bad',root)
    def test_ci_producer_rejects_other_event_branch_and_duplicates(self):
        root=self.fixture();folder,pin=self.buildset(root,'Windows');files=[*folder.glob('*.whl'),*folder.glob('*.tar.gz')]
        env=dict(GITHUB_SHA=COMMIT,GITHUB_EVENT_NAME='push',GITHUB_REF='refs/heads/main',GITHUB_WORKFLOW_REF=gate.REPOSITORY+'/.github/workflows/test.yml@refs/heads/main',GITHUB_REPOSITORY=gate.REPOSITORY,GITHUB_RUN_ID='123',GITHUB_RUN_ATTEMPT='1',RUNNER_OS='Windows')
        with self.assertRaises(ValueError):producer.manifest(env,files,root)
        env['GITHUB_EVENT_NAME']='workflow_dispatch';env['GITHUB_REF']='refs/heads/other'
        with self.assertRaises(ValueError):producer.manifest(env,files,root)
        env['GITHUB_REF']='refs/heads/main'
        with self.assertRaises(ValueError):producer.manifest(env,[*files,files[0]],root)

class Workflow(unittest.TestCase):
    def test_exact_isolated_upload_guard_accepts_only_reviewed_test_bundle(self):
        workflow=yaml.load((ROOT/'.github/workflows/publish.yml').read_text('utf8'),Loader=yaml.BaseLoader)
        code=workflow['jobs']['publish-testpypi']['steps'][-2]['run']
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);packages=folder/'reviewed/packages';packages.mkdir(parents=True)
            names=['morphtree-2.1.0-py3-none-win_amd64.whl','morphtree-2.1.0-py3-none-manylinux_2_17_x86_64.whl','morphtree-2.1.0.tar.gz']
            for name in names:(packages/name).write_bytes(name.encode())
            receipt=dict(schema='MORPHTREE_TESTPYPI_APPROVAL_1',repository=gate.REPOSITORY,commit=COMMIT,build_run_id=123,build_run_attempt=1,target='testpypi',version='2.1.0',repository_url=gate.TESTPYPI,files={name:gate.sha(packages/name) for name in names})
            path=folder/'reviewed/APPROVED.json';path.write_text(json.dumps(receipt),encoding='utf8')
            env=dict(REQUESTED='true',APPROVAL=gate.CONFIRMATION,APPROVED_MANIFEST_SHA256=gate.sha(path),GITHUB_REPOSITORY=gate.REPOSITORY,GITHUB_SHA=COMMIT,REVIEWED_COMMIT=COMMIT,REVIEWED_RUN='123',REVIEWED_ATTEMPT='1')
            def execute(updates=None,error=404):
                class Opener:
                    def open(self,*args,**kwargs):
                        if error is None:return io.BytesIO(b'{}')
                        raise urllib.error.HTTPError('https://test.pypi.org/',error,'fixture',{},None)
                previous=os.getcwd()
                try:
                    os.chdir(folder)
                    with patch.dict(os.environ,{**env,**(updates or {})}),patch('urllib.request.build_opener',return_value=Opener()):exec(compile(code,'exact-publish.yml-guard','exec'),{})
                finally:os.chdir(previous)
            execute()
            for updates in ({'REQUESTED':'false'},{'APPROVAL':''},{'APPROVED_MANIFEST_SHA256':'0'*64},{'GITHUB_SHA':'b'*40},{'REVIEWED_ATTEMPT':'2'}):
                with self.subTest(updates=updates),self.assertRaises(RuntimeError):execute(updates)
            for error in (None,403,429,500):
                with self.subTest(index=error),self.assertRaises(RuntimeError):execute(error=error)
            (packages/names[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):execute()
            (packages/names[0]).write_bytes(names[0].encode())
            (packages/'extra').write_bytes(b'x')
            with self.assertRaisesRegex(RuntimeError,'membership'):execute()
            (packages/'extra').unlink()
            receipt['repository_url']='https://upload.pypi.org/legacy/';path.write_text(json.dumps(receipt));env['APPROVED_MANIFEST_SHA256']=gate.sha(path)
            with self.assertRaisesRegex(RuntimeError,'PyPI is blocked'):execute()
    def test_oidc_job_is_test_only_isolated_and_manual(self):
        workflow=yaml.load((ROOT/'.github/workflows/publish.yml').read_text('utf8'),Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow['on']),{'workflow_dispatch'})
        inputs=workflow['on']['workflow_dispatch']['inputs']
        self.assertNotIn('target',inputs);self.assertEqual(inputs['publish_testpypi']['default'],'false')
        self.assertNotIn('id-token',workflow['permissions'])
        pre=workflow['jobs']['preflight'];upload=workflow['jobs']['publish-testpypi']
        self.assertNotIn('id-token',pre['permissions']);self.assertEqual(upload['permissions']['id-token'],'write')
        self.assertEqual(upload['environment']['name'],'testpypi');self.assertEqual(upload['needs'],'preflight')
        uses=[step['uses'] for step in upload['steps'] if 'uses' in step]
        self.assertFalse(any('checkout' in s or 'setup-python' in s for s in uses))
        action=upload['steps'][-1];self.assertEqual(action['uses'],'pypa/gh-action-pypi-publish@release/v1')
        self.assertEqual(action['with']['repository-url'],gate.TESTPYPI)
        self.assertEqual(action['with']['skip-existing'],'false');self.assertEqual(action['with']['attestations'],'true')
        self.assertNotIn('password',action['with'])
        for job in workflow['jobs'].values():
            for step in job['steps']:
                if step.get('shell')=='python':ast.parse(step['run'])
        text=(ROOT/'.github/workflows/publish.yml').read_text('utf8')
        self.assertNotIn('secrets.',text);self.assertNotIn('twine upload',text)
        handoff=(ROOT/'docs/LOCAL_HANDOFF.md').read_text('utf8')
        self.assertIn('PUBLISHING.md',handoff);self.assertIn('pypi_upload_enabled=false',handoff)
        self.assertNotIn('fails closed while RELEASE_STATUS.json',handoff)
    def test_provenance_emitted_after_metadata_and_tests(self):
        workflow=yaml.load((ROOT/'.github/workflows/test.yml').read_text('utf8'),Loader=yaml.BaseLoader)
        job=workflow['jobs']['scoped-zero-fit'];self.assertEqual(job['name'],'zero-fit-${{ matrix.os }}-${{ matrix.python }}')
        steps=job['steps'];position=next(i for i,s in enumerate(steps) if 'write_build_provenance.py' in s.get('run',''))
        self.assertTrue(any('unittest discover -s tools/tests' in s.get('run','') for s in steps[:position]))
        self.assertTrue(any('twine' in s.get('run','') and '--strict' in s['run'] for s in steps[:position]))
        self.assertIn('upload-artifact',steps[position+1]['uses'])

if __name__=='__main__':unittest.main()
