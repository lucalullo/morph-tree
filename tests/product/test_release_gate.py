"""Offline release blocker contracts; synthetic decisions never grant training."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('release_gate_under_test', Path(__file__).resolve().parents[2] / 'tools/release_gate.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


class ReleaseGateContracts(unittest.TestCase):
    def policy(self):
        return dict(schema='morphtree-release-qualification-v1', version='2.0.0',
            required_windows_python=['3.12','3.13','3.14'], release_ready=True,
            critical_open_issues=[], experimental_python=['3.15'], supported_platforms=dict(G.SUPPORTED_PLATFORMS),
            Linux_3_12_3={k:'PASS' for k in ('wheel','sdist','LIVE_three_tasks','integrity','normal_fresh_pip_install')}, windows={v:{k:'PASS' for k in ('wheel','sdist','LIVE_three_tasks','integrity','normal_fresh_pip_install')}
                for v in ('3.12','3.13','3.14')})

    def run_policy(self, value, *, version='2.0.0', python_spec='>=3.12,<3.15'):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'docs').mkdir()
            (root/'docs/QUALIFICATION_2_0.json').write_text(json.dumps(value),encoding='utf8')
            (root/'pyproject.toml').write_text('[project]\nversion="'+version+'"\nrequires-python="'+python_spec+'"\n',encoding='utf8')
            with patch.object(G,'ROOT',root):
                return G.qualification_gate()

    def test_incomplete_decision_cannot_publish(self):
        value=self.policy();value['release_ready']=False
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.run_policy(value)

    def test_missing_version_or_critical_issue_blocks_publication(self):
        value=self.policy();value['windows']['3.14']['LIVE_three_tasks']='MISSING'
        with self.assertRaisesRegex(ValueError,'3.14'):
            self.run_policy(value)
        value=self.policy();value['critical_open_issues']=['external blocker']
        with self.assertRaisesRegex(ValueError,'Critical'):
            self.run_policy(value)
        value=self.policy();value['windows']['3.14']['normal_fresh_pip_install']='FAIL'
        with self.assertRaisesRegex(ValueError,'3.14'):
            self.run_policy(value)

    def test_complete_synthetic_decision_is_read_without_network(self):
        value=self.policy()
        with patch.object(G,'gh_api',side_effect=AssertionError('No network in offline gate')):
            self.assertEqual(self.run_policy(value),value)

    def test_positive_decision_cannot_publish_narrow_or_unqualified_metadata(self):
        for python_spec in ('>=3.12,<3.13', '==3.15.*', '>=3.12', '>=3.12,<3.16', '>=3.12,<3.17'):
            with self.subTest(python_spec=python_spec), self.assertRaisesRegex(ValueError,'Release metadata'):
                self.run_policy(self.policy(),python_spec=python_spec)

    def test_missing_declared_linux_profile_blocks_readiness(self):
        value=self.policy();value['Linux_3_12_3']['sdist']='MISSING'
        with self.assertRaisesRegex(ValueError,'Linux/Python3.12'):
            self.run_policy(value)

    def test_unqualified_platform_or_315_promotion_is_rejected(self):
        for extra in ({'macOS':['3.12']}, {'Ubuntu24.04 WSL2 x86_64':['3.12','3.13']}):
            value=self.policy();value['supported_platforms'].update(extra)
            with self.assertRaisesRegex(ValueError,'platform scope'):
                self.run_policy(value)
        value=self.policy();value['experimental_python']=[]
        with self.assertRaisesRegex(ValueError,'platform scope'):
            self.run_policy(value)
        value=self.policy();value['normal_fresh_pip315_dependency_install']='FAIL_DEFAULT_SOURCE_BUILD'
        self.assertEqual(self.run_policy(value), value)  # Excluded315 is not a gate.

    def test_positive_decision_cannot_publish_another_package_version(self):
        with self.assertRaisesRegex(ValueError,'Release metadata'):
            self.run_policy(self.policy(),version='1.0.0')


class ReleaseArtifactContracts(unittest.TestCase):
    """Synthetic archives test promotion guards; no product fitting or network."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'docs').mkdir()
        (self.root/'dist').mkdir()
        policy = ReleaseGateContracts().policy()
        (self.root/'docs/QUALIFICATION_2_0.json').write_text(json.dumps(policy),encoding='utf8')
        (self.root/'pyproject.toml').write_text(
            '[project]\nname="morphtree"\nversion="2.0.0"\nrequires-python=">=3.12,<3.15"\n'
            'dependencies=["optuna==5.0.0"]\n',encoding='utf8')
        self.addCleanup(patch.stopall)
        patch.object(G,'ROOT',self.root).start()
        patch.object(G,'gh_api',side_effect=AssertionError('Offline only')).start()
        patch.dict(os.environ,{'GITHUB_RUN_ID':'123','GITHUB_SHA':'a'*40,
                               'GITHUB_REPOSITORY':'example/morph-tree'}).start()

    def artifacts(self, *, kind=None, version='2.0.0', python_spec='<3.15,>=3.12',
                  dependency='optuna==5.0.0', duplicate=False, purelib='true', native=False):
        for current in ('wheel','sdist'):
            altered = current == kind
            raw = ('Metadata-Version: 2.4\nName: morphtree\nVersion: '+(version if altered else '2.0.0')+
                   '\nRequires-Python: '+(python_spec if altered else '<3.15,>=3.12')+
                   '\nRequires-Dist: '+(dependency if altered else 'optuna==5.0.0')+'\n\n').encode()
            if altered and duplicate:
                raw = raw.replace(b'\n\n',b'\nVersion: 2.0.0\n\n')
            if current == 'wheel':
                with zipfile.ZipFile(self.root/'dist'/G.WHEEL,'w') as archive:
                    archive.writestr('morphtree-2.0.0.dist-info/METADATA',raw)
                    archive.writestr('morphtree-2.0.0.dist-info/WHEEL','Wheel-Version: 1.0\nRoot-Is-Purelib: '+purelib+'\nTag: py3-none-any\n\n')
                    if altered and native:
                        archive.writestr('morphtree/kernel.pyd',b'MZfixture only')
            else:
                with tarfile.open(self.root/'dist'/G.SDIST,'w:gz') as archive:
                    member=tarfile.TarInfo('morphtree-2.0.0/PKG-INFO');member.size=len(raw)
                    archive.addfile(member,io.BytesIO(raw))
                    if altered and native:
                        member=tarfile.TarInfo('morphtree-2.0.0/morphtree/kernel.c');member.size=6
                        archive.addfile(member,io.BytesIO(b'/*x*/\n'))

    def test_make_manifest_rejects_narrow_artifact_even_with_green_source(self):
        self.artifacts(kind='wheel',python_spec='>=3.12,<3.13')
        with self.assertRaisesRegex(ValueError,'artifact metadata'):
            G.make_manifest(self.root)
        self.assertFalse((self.root/'release-meta').exists())

    def test_both_formats_reject_wrong_version_dependency_or_duplicate_header(self):
        for kind in ('wheel','sdist'):
            for change in ({'version':'1.0.0'}, {'dependency':'optuna==4.0.0'}, {'duplicate':True}):
                with self.subTest(kind=kind,change=change):
                    self.artifacts(kind=kind,**change)
                    with self.assertRaisesRegex(ValueError,'artifact metadata'):
                        G.list_artifacts(self.root)

    def test_complete_archive_pair_can_be_verified_offline(self):
        self.artifacts()
        G.make_manifest(self.root)
        G.verify(self.root,expected_run_id='123')

    def test_direct_manifest_and_verify_cannot_bypass_no_go(self):
        self.artifacts()
        G.make_manifest(self.root)
        value=ReleaseGateContracts().policy();value['release_ready']=False
        (self.root/'docs/QUALIFICATION_2_0.json').write_text(json.dumps(value),encoding='utf8')
        for action in (G.make_manifest,G.verify):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError,'incomplete'):
                action(self.root)

    def test_verify_rejects_wrong_metadata_even_with_matching_manifest_hash(self):
        self.artifacts()
        G.make_manifest(self.root)
        self.artifacts(kind='sdist',python_spec='>=3.12,<3.13')
        path=self.root/'release-meta/provenance.json'
        value=json.loads(path.read_text('utf8'))
        source=self.root/'dist'/G.SDIST
        value['artifacts'][G.SDIST]={'sha256':G.check_hash(source),'bytes':source.stat().st_size}
        path.write_text(json.dumps(value),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'artifact metadata'):
            G.verify(self.root,expected_run_id='123')

    def test_release_wheel_requires_purelib_tag(self):
        self.artifacts(purelib='false')
        with self.assertRaisesRegex(ValueError,'Pure Python'):
            G.list_artifacts(self.root)

    def test_proprietary_native_payloads_are_rejected_in_both_formats(self):
        for kind in ('wheel','sdist'):
            self.artifacts(kind=kind,native=True)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError,'non-Python proprietary'):
                G.list_artifacts(self.root)
