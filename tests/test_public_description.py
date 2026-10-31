"""Installed Markdown description approval, without new training."""
import email.policy,hashlib,importlib.metadata,json,unittest
from email.parser import Parser
from pathlib import Path
import morphtree

class PublicDescription(unittest.TestCase):
    def contract(self):return json.loads((Path(morphtree.__file__).parent/'_experimental/manifest.json').read_text('utf8'))['public_description']
    def test_installed_description_is_approved(self):
        text=importlib.metadata.distribution('morphtree').read_text('METADATA')
        body=Parser(policy=email.policy.compat32).parsestr(text).get_payload()
        self.assertEqual(hashlib.sha256(body.replace('\r\n','\n').encode('utf8')).hexdigest(),self.contract()['normalized_description_sha256'])
    def test_explicit_markdown_content_type(self):
        self.assertEqual(importlib.metadata.metadata('morphtree')['Description-Content-Type'],'text/markdown')
    def test_source_approval_is_recorded(self):
        c=self.contract();self.assertEqual(c['source'],'README.md');self.assertEqual(len(c['README_sha256']),64);self.assertFalse(c['README_modified_by_agent'])
if __name__=='__main__':unittest.main()
