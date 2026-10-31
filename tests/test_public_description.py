"""Installed Markdown description integrity and authorized revision, without new training."""
import email.policy,hashlib,importlib.metadata,json,unittest
from email.parser import Parser
from pathlib import Path
import morphtree

class PublicDescription(unittest.TestCase):
    def contract(self):return json.loads((Path(morphtree.__file__).parent/'_experimental/manifest.json').read_text('utf8'))['public_description']
    def test_installed_description_matches_source_contract(self):
        text=importlib.metadata.distribution('morphtree').read_text('METADATA')
        body=Parser(policy=email.policy.compat32).parsestr(text).get_payload()
        self.assertEqual(hashlib.sha256(body.replace('\r\n','\n').encode('utf8')).hexdigest(),self.contract()['normalized_description_sha256'])
    def test_explicit_markdown_content_type(self):
        self.assertEqual(importlib.metadata.metadata('morphtree')['Description-Content-Type'],'text/markdown')
    def test_source_revision_authority_is_recorded(self):
        c=self.contract()
        self.assertEqual(c['source'],'README.md')
        self.assertRegex(c['README_sha256'],r'^[0-9a-f]{64}$')
        self.assertTrue(c['README_modified_by_agent'])
        self.assertEqual(c['source_revision_authorization'],'EXPLICIT_OWNER_DIRECTIVE')
        self.assertEqual(c['authorization_sha256'],'da902c3c4bc4ca321a2e94e5e4513dc1f714935fa17c1d602c9e08e2d059f0b3')
        self.assertTrue(c['source_owner_review_pending'])
if __name__=='__main__':unittest.main()
