"""Validate the reviewed source inventory without importing MorphTree."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.native-builds', 'dist', 'build', '.venv', 'env'}
def inventory():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file()
            and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
            and p.relative_to(ROOT).as_posix() != 'SOURCE_MANIFEST.json'
            and p.suffix not in ('.pyc', '.pyo')}
def check():
    expected = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text('utf8'))['files']
    actual = inventory()
    if actual != expected:
        raise RuntimeError('Source inventory drift: '+str(sorted(set(actual)^set(expected)))+
                           '; changed='+str([k for k in actual.keys()&expected.keys() if actual[k]!=expected[k]]))
    required = ['LICENSE', 'pyproject.toml', 'hatch_build.py', 'tools/build_native.py',
                'python/morphtree/_native/src/histogram.c', 'python/morphtree/_native/src/traversal.c']
    assert all(k in actual for k in required)
    print(json.dumps({'status':'PASS_SOURCE_INVENTORY', 'files':len(actual), 'model_fits':0}))
    return actual
if __name__ == '__main__': check()
