"""Offline official-index protocol regression tests. No real APIs or uploads."""
import ast,copy,io,json,os,socket,tempfile,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch
import yaml
import test_publishing as previous

ROOT=previous.ROOT
gate=previous.load('pypi_publication_checks',ROOT/'tools/publish_pypi_reviewed.py')
COMMIT=previous.COMMIT

class Production(unittest.TestCase):
    def setUp(self):
        self.base=previous.Publishing('test_success_origin');self.base.setUp()
        self.environment=dict(name='pypi',protection_rules=[dict(type='required_reviewers',reviewers=[dict(type='User',reviewer=dict(login='lucalullo'))])],deployment_branch_policy=dict(protected_branches=False,custom_branch_policies=True))
        self.env=dict(GITHUB_REPOSITORY=gate.shared.REPOSITORY,GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_REF='refs/heads/main',GITHUB_SHA=COMMIT,GITHUB_WORKFLOW_REF=gate.shared.REPOSITORY+'/.github/workflows/publish-pypi.yml@refs/heads/main')
    def tearDown(self):self.base.tearDown()
    def origin(self):
        values=iter([self.base.run,dict(total_count=6,jobs=self.base.jobs),dict(total_count=6,artifacts=self.base.artifacts),self.environment])
        return gate.inspect_origin(COMMIT,'123',1,'fake',lambda *args:next(values))
    def fixture(self):
        root=self.base.fixture();path=root/'docs/RELEASE_STATUS.json';status=gate.shared.read(path)
        status['pypi_workflow']=dict(prepared=True,environment='pypi',runtime_dispatch_confirmation=gate.CONFIRMATION,publication_approved_by_preparation=False)
        path.write_text(json.dumps(status))
        baseline=['README.md','pyproject.toml']
        (root/'docs/PYPI_RUNTIME_REFERENCE.json').write_text(json.dumps(dict(build_files=baseline,files={n:gate.shared.sha(root/n) for n in baseline})))
        (root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(files={p.relative_to(root).as_posix():gate.shared.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='SOURCE_MANIFEST.json'})))
        return root
    def indices(self,url,*args):
        if url==gate.TEST_JSON:return dict(info=dict(name='morphtree',version='2.1.0'),urls=[dict(digests=dict(sha256='f'*64),yanked=False)])
        if url==gate.PYPI_JSON:raise urllib.error.HTTPError(url,404,'fixture',{},None)
        raise AssertionError('Unexpected verification URL')
    def test_exact_protected_origin_and_context(self):
        origin=self.origin();self.assertEqual(origin['protected_environment'],'pypi');self.assertTrue(origin['pypi_policy_verified'])
        gate.require_context(COMMIT,self.env)
    def test_old_commit_workflow_branch_or_fork_rejected(self):
        for key,value in [('GITHUB_SHA','b'*40),('GITHUB_WORKFLOW_REF',gate.shared.REPOSITORY+'/.github/workflows/publish.yml@refs/heads/main'),('GITHUB_REF','refs/tags/v2.1.0'),('GITHUB_EVENT_NAME','push'),('GITHUB_REPOSITORY','other/fork')]:
            with self.subTest(key=key),self.assertRaises(ValueError):gate.require_context(COMMIT,{**self.env,key:value})
    def test_wrong_environment_reviewer_rejected(self):
        original=copy.deepcopy(self.environment)
        for field in ('name','reviewer'):
            env=copy.deepcopy(original)
            if field=='name':env['name']='testpypi'
            else:env['protection_rules'][0]['reviewers'][0]['reviewer']['login']='another-user'
            self.environment=env
            with self.assertRaises(ValueError):self.origin()
    def test_unrestricted_environment_and_non_main_workflow_rejected(self):
        for policy in (None,dict(protected_branches=False,custom_branch_policies=False),dict(protected_branches=True,custom_branch_policies=False)):
            self.environment['deployment_branch_policy']=policy
            with self.assertRaises(ValueError):gate.require_environment(self.environment)
        for branch in ('refs/tags/main','refs/heads/dev','refs/heads/main-other'):
            with self.assertRaises(ValueError):gate.require_context(COMMIT,{**self.env,'GITHUB_REF':branch})
    def test_old_CI_commit_and_failed_or_skipped_jobs_rejected(self):
        self.base.run['head_sha']='b'*40
        with self.assertRaises(ValueError):self.origin()
        self.base.run['head_sha']=COMMIT
        for conclusion in ('skipped','cancelled','failure'):
            self.base.jobs[0]['conclusion']=conclusion
            with self.assertRaises(ValueError):self.origin()
    def test_prepared_policy_keeps_legacy_publication_flags_false(self):
        root=self.fixture();gate.require_policy('2.1.0',root)
        self.assertFalse(gate.shared.read(root/'docs/RELEASE_STATUS.json')['publication_approved'])
        with self.assertRaises(ValueError):gate.require_policy('2.2.0',root)
        with self.assertRaises(ValueError):gate.shared.require_policy('pypi','2.1.0',root)
    def test_changed_runtime_or_missing_policy_rejected(self):
        root=self.fixture();(root/'README.md').write_text('changed')
        with self.assertRaises(ValueError):gate.require_policy('2.1.0',root)
        (root/'docs/RELEASE_STATUS.json').write_text('{}')
        with self.assertRaises((ValueError,KeyError)):gate.require_policy('2.1.0',root)
    def test_existing_test_release_and_absent_production_only(self):
        self.assertTrue(gate.require_indices(self.indices)['testpypi_release_verified'])
        with self.assertRaises(ValueError):gate.require_indices(lambda url:self.indices(gate.TEST_JSON))
        for status in (401,403,429,500):
            def failed(url):
                if url==gate.TEST_JSON:return self.indices(url)
                raise urllib.error.HTTPError(url,status,'fixture',{},None)
            with self.assertRaises(urllib.error.HTTPError):gate.require_indices(failed)
    def test_missing_wrong_version_or_yanked_test_release_rejected(self):
        for release in (dict(info=dict(name='other',version='2.1.0'),urls=[{}]),dict(info=dict(name='morphtree',version='2.0.0'),urls=[{}]),dict(info=dict(name='morphtree',version='2.1.0'),urls=[]),dict(info=dict(name='morphtree',version='2.1.0'),urls=[dict(yanked=True,digests=dict(sha256='f'*64))])):
            with self.subTest(release=release),self.assertRaises(ValueError):gate.require_indices(lambda url:release)
        def absent(url):raise urllib.error.HTTPError(url,404,'fixture',{},None)
        with self.assertRaises(urllib.error.HTTPError):gate.require_indices(absent)
    def test_no_redirect_credentials_or_unexpected_service(self):
        with self.assertRaises(ValueError):gate.get_json(gate.PYPI_JSON,'must-not-forward')
        with self.assertRaises(ValueError):gate.get_json('https://example.invalid/')
        with self.assertRaises(ValueError):gate.shared.NoRedirect().redirect_request(None,None,None,None,None,None)
    def test_real_bundle_logic_production_receipt_and_tampered_build(self):
        root=self.fixture();windows,wsha=self.base.buildset(root,'Windows');linux,lsha=self.base.buildset(root,'Linux')
        value=gate.prepare_bundle(windows,linux,wsha,lsha,self.origin(),self.base.root/'production',root,self.indices)
        self.assertEqual(value['target'],'pypi');self.assertEqual(value['repository_url'],gate.PYPI);self.assertTrue(value['testpypi_release_verified'])
        self.assertEqual(len(value['files']),3)
        with self.assertRaises(ValueError):gate.prepare_bundle(windows,linux,'0'*64,lsha,self.origin(),self.base.root/'bad',root,self.indices)
        wrong=self.origin();wrong['protected_environment']='testpypi'
        with self.assertRaises(ValueError):gate.prepare_bundle(windows,linux,wsha,lsha,wrong,self.base.root/'wrong',root,self.indices)

class Workflow(unittest.TestCase):
    def workflow(self):return yaml.load((ROOT/'.github/workflows/publish-pypi.yml').read_text('utf8'),Loader=yaml.BaseLoader)
    def test_production_is_manual_isolated_and_environment_protected(self):
        workflow=self.workflow();self.assertEqual(set(workflow['on']),{'workflow_dispatch'})
        inputs=workflow['on']['workflow_dispatch']['inputs'];self.assertEqual(inputs['publish_pypi']['default'],'false');self.assertNotIn('target',inputs)
        for name in ('reviewed_commit','reviewed_build_run','reviewed_build_attempt','windows_manifest_sha256','linux_manifest_sha256'):self.assertEqual(inputs[name]['required'],'true')
        pre=workflow['jobs']['preflight'];upload=workflow['jobs']['publish-pypi']
        self.assertNotIn('id-token',workflow['permissions']);self.assertNotIn('id-token',pre['permissions'])
        self.assertEqual(upload['permissions']['id-token'],'write');self.assertEqual(upload['environment']['name'],'pypi')
        self.assertEqual(upload['needs'],'preflight');self.assertIn(gate.CONFIRMATION,upload['if'])
        action=upload['steps'][-1];self.assertEqual(action['uses'],'pypa/gh-action-pypi-publish@release/v1');self.assertEqual(action['with']['repository-url'],gate.PYPI)
        self.assertNotIn('password',action['with']);self.assertEqual(action['with']['skip-existing'],'false')
        self.assertFalse(any('checkout' in s.get('uses','') or 'setup-python' in s.get('uses','') or 'pip install' in s.get('run','') for s in upload['steps']))
        for job in workflow['jobs'].values():
            for step in job['steps']:
                if step.get('shell')=='python':ast.parse(step['run'])
        self.assertNotIn('secrets.',(ROOT/'.github/workflows/publish-pypi.yml').read_text('utf8'))
    def test_exact_OIDC_guard_accepts_only_same_reviewed_production_bundle(self):
        code=self.workflow()['jobs']['publish-pypi']['steps'][-2]['run']
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);packages=folder/'reviewed/packages';packages.mkdir(parents=True)
            names=['morphtree-2.1.0-py3-none-win_amd64.whl','morphtree-2.1.0-py3-none-manylinux_2_17_x86_64.whl','morphtree-2.1.0.tar.gz']
            for name in names:(packages/name).write_bytes(name.encode())
            receipt=dict(schema='MORPHTREE_PYPI_APPROVAL_1',repository=gate.shared.REPOSITORY,commit=COMMIT,build_run_id=123,build_run_attempt=1,target='pypi',version='2.1.0',repository_url=gate.PYPI,testpypi_release_verified=True,publication_workflow='.github/workflows/publish-pypi.yml',files={n:gate.shared.sha(packages/n) for n in names})
            path=folder/'reviewed/APPROVED.json';path.write_text(json.dumps(receipt))
            env=dict(REQUESTED='true',APPROVAL=gate.CONFIRMATION,APPROVED_MANIFEST_SHA256=gate.shared.sha(path),GITHUB_REPOSITORY=gate.shared.REPOSITORY,GITHUB_SHA=COMMIT,REVIEWED_COMMIT=COMMIT,REVIEWED_RUN='123',REVIEWED_ATTEMPT='1',GITHUB_WORKFLOW_REF=gate.shared.REPOSITORY+'/.github/workflows/publish-pypi.yml@refs/heads/main',GITHUB_REF='refs/heads/main',GITHUB_EVENT_NAME='workflow_dispatch')
            def execute(updates=None,index=404):
                class Opener:
                    def open(self,*args,**kwargs):
                        if index is None:return io.BytesIO(b'{}')
                        raise urllib.error.HTTPError(gate.PYPI_JSON,index,'fixture',{},None)
                previous_dir=os.getcwd()
                try:
                    os.chdir(folder)
                    with patch.dict(os.environ,{**env,**(updates or {})}),patch('urllib.request.build_opener',return_value=Opener()):exec(compile(code,'exact-production-guard','exec'),{})
                finally:os.chdir(previous_dir)
            execute()
            for change in ({'REQUESTED':'false'},{'APPROVAL':previous.gate.CONFIRMATION},{'GITHUB_SHA':'b'*40},{'REVIEWED_ATTEMPT':'2'},{'APPROVED_MANIFEST_SHA256':'0'*64},{'GITHUB_REF':'refs/heads/dev'}):
                with self.subTest(change=change),self.assertRaises(RuntimeError):execute(change)
            for index in (None,403,429,500):
                with self.assertRaises(RuntimeError):execute(index=index)
            (packages/names[0]).write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):execute()
            (packages/names[0]).write_bytes(names[0].encode());(packages/'extra').write_bytes(b'x')
            with self.assertRaisesRegex(RuntimeError,'membership'):execute()
            (packages/'extra').unlink();receipt['testpypi_release_verified']=False;path.write_text(json.dumps(receipt));env['APPROVED_MANIFEST_SHA256']=gate.shared.sha(path)
            with self.assertRaisesRegex(RuntimeError,'TestPyPI release'):execute()

if __name__=='__main__':unittest.main()
