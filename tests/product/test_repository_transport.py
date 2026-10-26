"""Zero-fit source-transport contracts. No Git repository or remote is created."""
import hashlib,importlib.util,io,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

source=Path(__file__).resolve().parents[2]/'tools/check_repository.py'
spec=importlib.util.spec_from_file_location('repository_snapshot_checker',source)
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)

class RepositoryTransportTests(unittest.TestCase):
 def setUp(self):
  self.files={'code.py':b'fixture only\n','research/negative.md':b'FAIL preserved\n'}
  self.expected={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in self.files.items()}
 def archive(self,rows):
  data=io.BytesIO()
  with tarfile.open(fileobj=data,mode='w') as t:
   for name,body,kind in rows:
    info=tarfile.TarInfo(name);info.type=kind;info.size=len(body) if kind==tarfile.REGTYPE else 0
    if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):info.linkname='code.py'
    t.addfile(info,io.BytesIO(body) if kind==tarfile.REGTYPE else None)
  data.seek(0);return data
 def rows(self):return [(n,b,tarfile.REGTYPE) for n,b in self.files.items()]
 def test_valid_exact_archive(self):
  self.assertEqual(checker.verify_archive(self.archive(self.rows()),self.expected),2)
 def test_changed_committed_byte(self):
  rows=self.rows();rows[0]=('code.py',b'fixture evil\n',tarfile.REGTYPE)
  with self.assertRaises(ValueError):checker.verify_archive(self.archive(rows),self.expected)
 def test_missing_research_file(self):
  with self.assertRaises(ValueError):checker.verify_archive(self.archive(self.rows()[:1]),self.expected)
 def test_extra_private_file(self):
  with self.assertRaises(ValueError):checker.verify_archive(self.archive(self.rows()+[('.research-local/private.txt',b'FIXTURE',tarfile.REGTYPE)]),self.expected)
 def test_duplicate_file(self):
  with self.assertRaises(ValueError):checker.verify_archive(self.archive(self.rows()+self.rows()[:1]),self.expected)
 def test_links_and_unknown_directory(self):
  for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
   with self.subTest(kind=kind),self.assertRaises(ValueError):checker.verify_archive(self.archive(self.rows()+[('unexpected',b'',kind)]),self.expected)
 def test_traversal(self):
  with self.assertRaises(ValueError):checker.verify_archive(self.archive([('../outside',b'FIXTURE',tarfile.REGTYPE)]),self.expected)
 def test_invalid_commit_and_absent_repo_are_readonly_rejections(self):
  with patch.object(checker.subprocess,'run',side_effect=AssertionError('No Git calls in these contracts')):
   with self.assertRaises(ValueError):checker.verify_git_head('not-a-commit')
   missing=source.parent.parent/'__not_created_repo_fixture__'
   self.assertFalse(missing.exists())
   with patch.object(checker,'ROOT',missing):
    with self.assertRaises(ValueError):checker.verify_git_head()
    self.assertFalse(missing.exists())

 def test_pure_python_policy_rejects_native_sources_and_disguised_binary(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);package=root/'morphtree';package.mkdir()
   (package/'engine.py').write_text('def identity(x): return x\n',encoding='utf8')
   self.assertEqual(checker.verify_python_engine(root),1)
   for name,body in (('split.c',b'int x;'),('split.pyd',b'binary fixture'),('hidden.dat',b'\x7fELFfixture')):
    path=package/name;path.write_bytes(body)
    with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Pure Python'):
     checker.verify_python_engine(root)
    path.unlink()

 def test_pure_python_policy_rejects_jit_and_custom_library_loaders(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);package=root/'morphtree';package.mkdir();path=package/'engine.py'
   for code in ('import numba\n','import importlib\nimportlib.import_module("numba")\n','import ctypes\nctypes.CDLL("proprietary.dll")\n'):
    path.write_text(code,encoding='utf8')
    with self.subTest(code=code),self.assertRaisesRegex(ValueError,'Pure Python'):
     checker.verify_python_engine(root)

if __name__=='__main__':unittest.main(verbosity=2)
