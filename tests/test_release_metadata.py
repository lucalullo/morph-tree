"""Installed identity and scoped evidence; these tests perform no training."""
import importlib.metadata
import json
import unittest
from pathlib import Path
import morphtree
from morphtree.api import PUBLIC_USER_RUNTIME
from morphtree._native.loader import runtime_info
from morphtree.scalable import verify_source_identity

class ReleaseMetadata(unittest.TestCase):
    def manifest(self):
        return json.loads((Path(morphtree.__file__).parent/'_experimental/manifest.json').read_text('utf8'))

    def test_distribution_package_api_and_manifest_version_agree(self):
        self.assertEqual(importlib.metadata.version('morphtree'), '2.1.0')
        self.assertEqual(morphtree.__version__, PUBLIC_USER_RUNTIME)
        self.assertEqual(PUBLIC_USER_RUNTIME, self.manifest()['version'])
        verify_source_identity()

    def test_backend_policy_matches_loaded_native_library(self):
        self.assertEqual(morphtree.__runtime__['backend'], 'automatic-native')
        self.assertEqual(self.manifest()['backend'], 'automatic-native')
        info=runtime_info()
        self.assertEqual(info['ABI'], 1)
        self.assertEqual(len(info['binary_sha256']), 64)
        self.assertFalse(info['runtime_compilation'])

    def test_qualification_has_scope_and_does_not_grant_publication(self):
        current=self.manifest()['native_lab']
        self.assertTrue(current['whole_pipeline_native_qualified'])
        self.assertEqual(len(current['fresh_RC_profiles']),2)
        self.assertEqual(current['evidence_campaigns'],['VERSION121','VERSION122'])
        self.assertEqual(current['training_distribution_version'],'2.1.0')
        self.assertTrue(current['version_approved'])
        self.assertFalse(current['public_release'])
        self.assertFalse(current['publication_approved'])
        self.assertEqual(len(current['qualified_profiles']),3)
        self.assertIn('million-row training',current['unqualified'])
        self.assertIn('Python 3.15 ordinary install',current['unqualified'])
    def test_historical_provenance_is_retained_and_labelled(self):
        old=self.manifest()['historical_provenance']
        self.assertEqual(old['original_python_reference_version'],'2.1.0')
        self.assertEqual(old['previous_native_reference_version'],'2.1.0')
        self.assertFalse(old['learning_mathematics_changed'])
        self.assertFalse(self.manifest()['capacity_profile']['new_default_adopted'])
