"""Portable zero-fit check of the installed package, outside the source tree."""
import hashlib,json,os,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import morphtree
PACKAGE=Path(morphtree.__file__).resolve().parent
assert PACKAGE.is_relative_to(Path(sys.prefix).resolve()) and not PACKAGE.is_relative_to(ROOT)
from morphtree.booster import BinaryHistogramBooster
from morphtree.tree import HistogramTree
from morphtree.scalable import verify_source_identity
from morphtree._native.loader import runtime_info

def inventory(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}

def main():
    previous=Path.cwd()
    source_before=inventory(ROOT)
    with tempfile.TemporaryDirectory(prefix='morphtree-public-check-') as d:
        work=Path(d)
        for p in (ROOT/'python/tests/product').glob('test_*.py'):shutil.copyfile(p,work/p.name)
        for p in (ROOT/'tests').glob('test_*.py'):shutil.copyfile(p,work/p.name)
        sys.path.insert(0,str(work));os.chdir(work)
        work_before=inventory(work)
        counts=[]
        try:
            with patch.object(BinaryHistogramBooster,'fit',side_effect=AssertionError('No learner fits allowed')),patch.object(HistogramTree,'fit',side_effect=AssertionError('No tree fits allowed')):
                verify_source_identity()
                for repetition in (1,2):
                    # unittest consumes suites; rediscover tests, retaining the
                    # same directory and loaded native runtime/cache.
                    suite=unittest.defaultTestLoader.discover(str(work))
                    expected=suite.countTestCases()
                    assert expected>=36,'Required test suite missing'
                    result=unittest.TextTestRunner(verbosity=2).run(suite)
                    assert result.wasSuccessful() and result.testsRun==expected and not result.skipped
                    assert inventory(work)==work_before,'Tests wrote or changed working-directory files'
                    assert inventory(ROOT)==source_before,'Tests changed the source repository'
                    counts.append(result.testsRun)
                assert counts[0]==counts[1]
                info=runtime_info()
            assert all(not Path(m.__file__).resolve().is_relative_to(ROOT) for n,m in tuple(sys.modules.items()) if n.startswith('morphtree') and getattr(m,'__file__',None))
            print(json.dumps({'status':'PASS_INSTALLED_ZERO_FIT','Python':sys.version,
                              'tests':expected,'passes':len(counts),'test_executions':sum(counts),
                              'same_directory_repeated':True,'working_files_unchanged':True,
                              'source_files_unchanged':True,'package':str(PACKAGE),'native':info,'model_fits':0}))
        finally:os.chdir(previous)
if __name__=='__main__':main()
