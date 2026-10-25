"""Finite zero-fit installed-artifact check for separately approved private CI."""
import argparse,base64,csv,hashlib,importlib.metadata,io,json,sys,zipfile
from pathlib import Path

def check(wheel,expected,version):
 if not sys.flags.isolated:raise ValueError('Run with python -I -B')
 wheel=Path(wheel);actual=hashlib.sha256(wheel.read_bytes()).hexdigest()
 if actual!=expected:raise ValueError('Externally pinned wheel SHA mismatch')
 with zipfile.ZipFile(wheel) as z:
  names=z.namelist()
  if len(names)!=len(set(names)) or len(names)>100:raise ValueError('Unexpected wheel inventory')
  if any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n for n in names):raise ValueError('Unsafe wheel member')
  blobs={n:z.read(n) for n in names}
  record=next(n for n in names if n.endswith('.dist-info/RECORD'))
  rows=list(csv.reader(io.StringIO(blobs[record].decode('utf8'))))
  if {r[0] for r in rows}!=set(names) or len(rows)!=len(names):raise ValueError('Incomplete/duplicate wheel RECORD')
  for n,h,size in rows:
   if n==record:
    if h or size:raise ValueError('RECORD self-entry must be unhashed')
    continue
   want='sha256='+base64.urlsafe_b64encode(hashlib.sha256(blobs[n]).digest()).decode().rstrip('=')
   if h!=want or int(size)!=len(blobs[n]):raise ValueError('Wheel RECORD mismatch')
 import morphtree,optuna
 from unittest.mock import patch
 from morphtree import booster,tree
 if morphtree.__version__!=version or importlib.metadata.version('morphtree')!=version:raise ValueError('Distribution/import version mismatch')
 if optuna.__version__!='5.0.0':raise ValueError('Unqualified Optuna version')
 package=Path(morphtree.__file__).resolve().parent
 if not package.is_relative_to(Path(sys.prefix).resolve()):raise ValueError('Not an isolated installed package')
 manifest=json.loads(blobs['morphtree/_experimental/manifest.json'])
 for n,h in manifest['files'].items():
  raw=(package/n).read_bytes()
  if raw!=blobs['morphtree/'+n] or hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Installed/wheel content mismatch')
 if (package/'_experimental/manifest.json').read_bytes()!=blobs['morphtree/_experimental/manifest.json']:raise ValueError('Changed installed manifest')
 attempts=[]
 def forbidden(*a,**k):attempts.append('learning');raise AssertionError('No learning permitted in zero-fit CI')
 from contextlib import ExitStack
 with ExitStack() as stack:
  for name in ('BinaryHistogramBooster','MulticlassHistogramBooster','RegressionHistogramBooster'):
   typ=getattr(booster,name,None)
   if typ is not None:stack.enter_context(patch.object(typ,'fit',forbidden))
  stack.enter_context(patch.object(tree.HistogramTree,'fit',forbidden))
  stack.enter_context(patch.object(optuna,'create_study',forbidden))
  for task in ('binary','multiclass','regression'):
   m=morphtree.MorphTree(task=task,random_state=42,verbosity=0)
   if not m._evolving_program or m.tuning!='optuna':raise ValueError('Default evolution/tuning missing')
 if attempts:raise ValueError('Unexpected learning attempt')
 if any(n=='tools' or n.startswith('tools.') for n in sys.modules):raise ValueError('Checkout namespace imported')
 return {'status':'PASS_ZERO_FIT_INSTALLED_ARTIFACT','version':version,'wheel_sha256':actual,'manifest_files':len(manifest['files']),'native_fits':0,'trials':0,'scope':'Import/metadata/RECORD/native defaults only;not LIVE,OS matrix or performance proof.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wheel',required=True);p.add_argument('--sha256',required=True);p.add_argument('--version',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 out=Path(a.out)
 if out.exists():raise FileExistsError('Never replace a completed check')
 value=check(a.wheel,a.sha256,a.version);out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))
if __name__=='__main__':main()
