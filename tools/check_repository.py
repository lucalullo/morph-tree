"""Verify the exact local source snapshot without importing the learner."""
import argparse,ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify_python_engine(root):
 """Reject proprietary native sources/binaries/JIT; do not import the engine."""
 forbidden={'.c','.cc','.cpp','.cxx','.h','.hpp','.rs','.pyx','.pxd','.pyd','.so','.dll','.dylib','.a','.o','.obj','.lib','.exe','.wasm'}
 compiler_modules={'numba','llvmlite','Cython','cython','pyximport','cffi','cppyy','pybind11','maturin'}
 system_libraries={'_experimental/processes.py':('kernel32','psapi'),'_experimental/worker.py':('kernel32',)}
 count=0
 for path in (root/'morphtree').rglob('*'):
  if not path.is_file() or '__pycache__' in path.parts:continue
  name=path.relative_to(root/'morphtree').as_posix()
  if path.is_symlink() or any(s.lower() in forbidden for s in path.suffixes):raise ValueError('Pure Python engine policy: '+name)
  raw=path.read_bytes()
  if raw[:4] in (b'\x7fELF',b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf') or raw[:2]==b'MZ':raise ValueError('Pure Python engine policy: binary '+name)
  if path.suffix!='.py':continue
  count+=1
  for node in ast.walk(ast.parse(raw,filename=name)):
   modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
   if any(m.split('.')[0] in compiler_modules for m in modules):raise ValueError('Pure Python engine policy: compiler/JIT import in '+name)
   if isinstance(node,ast.Call):
    target=ast.unparse(node.func)
    if target in ('ctypes.CDLL','ctypes.PyDLL','ctypes.cdll.LoadLibrary'):raise ValueError('Pure Python engine policy: proprietary library loader in '+name)
    if target=='ctypes.WinDLL' and (not node.args or not isinstance(node.args[0],ast.Constant) or node.args[0].value not in system_libraries.get(name,())):raise ValueError('Pure Python engine policy: unqualified WinDLL in '+name)
    if (target.endswith('import_module') or target=='__import__') and node.args and isinstance(node.args[0],ast.Constant) and str(node.args[0].value).split('.')[0] in compiler_modules:raise ValueError('Pure Python engine policy: dynamic compiler import in '+name)
    if target.rsplit('.',1)[-1] in ('run','Popen','check_call','check_output','system') and node.args:
     command=node.args[0]
     first=command.elts[0] if isinstance(command,(ast.List,ast.Tuple)) and command.elts else command
     if isinstance(first,ast.Constant) and isinstance(first.value,str) and re.split(r'[\\/\s]',first.value)[-1].removesuffix('.exe') in {'gcc','g++','clang','clang++','cl','cc','c++','rustc','cargo','cmake','nvcc','cython'}:raise ValueError('Pure Python engine policy: external compiler in '+name)
 return count
def verify():
 python_modules=verify_python_engine(ROOT)
 m=json.loads((ROOT/'LOCAL_SOURCE_MANIFEST.json').read_text('utf8'))
 if m['schema']!='morphtree-local-github-source-v2' or m['remote_operations']!=0:raise ValueError('Unexpected repository manifest')
 if len(m['files'])>10000 or sum(e['bytes'] for e in m['files'].values())>400*1024**2:raise ValueError('Source inventory limit')
 files=set()
 for directory,dirs,names in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','.venv','.pytest_cache','build','dist'}]
  for n in names:
   if n.endswith(('.pyc','.pyo')):continue
   files.add((Path(directory)/n).relative_to(ROOT).as_posix())
 if files!=set(m['files'])|{'LOCAL_SOURCE_MANIFEST.json','LOCAL_REPOSITORY_STATUS.md'}:raise ValueError('Unexpected/missing source files')
 for n,e in m['files'].items():
  p=ROOT/n
  if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Unsafe source path')
  b=p.read_bytes()
  if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('Changed source: '+n)
 b=(ROOT/'LOCAL_REPOSITORY_STATUS.md').read_bytes()
 if hashlib.sha256(b).hexdigest()!=m['status_sha256']:raise ValueError('Status changed')
 for n,h in m['product']['modules'].items():
  if m['files']['morphtree/'+n]['sha256']!=h:raise ValueError('Source/candidate mismatch')
 return {'status':'PASS_LOCAL_SOURCE_INTEGRITY','files':len(m['files']),'version':m['product']['version'],'release':'GO_LOCAL' if json.loads((ROOT/'docs/QUALIFICATION_2_0.json').read_text('utf8')).get('release_ready') is True else 'NO_GO','new_training':0,'pure_python_engine_modules':python_modules}

def verify_archive(stream,expected):
 """Read a Git tar stream without extracting it or trusting its filenames."""
 parents={p.as_posix() for n in expected for p in Path(n).parents if p.as_posix()!='.'}
 seen=set()
 with tarfile.open(fileobj=stream,mode='r|') as archive:
  for member in archive:
   name=member.name
   if member.isdir():
    if name not in parents:raise ValueError('Unexpected committed directory')
    continue
   if not member.isfile() or member.sparse:raise ValueError('Committed links/special files forbidden')
   if name not in expected or name in seen:raise ValueError('Missing/extra/duplicate committed file')
   e=expected[name]
   if member.size!=e['bytes']:raise ValueError('Committed file size mismatch: '+name)
   digest=hashlib.sha256();total=0
   with archive.extractfile(member) as data:
    for block in iter(lambda:data.read(65536),b''):digest.update(block);total+=len(block)
   if total!=e['bytes'] or digest.hexdigest()!=e['sha256']:raise ValueError('Committed file hash mismatch: '+name)
   seen.add(name)
 if seen!=set(expected):raise ValueError('Missing committed source files')
 return len(seen)

def verify_git_head(expected_commit=None):
 """Owner/CI-only read of an already-created local repository. No Git writes."""
 if expected_commit is not None and not re.fullmatch('[0-9a-f]{40}',expected_commit):raise ValueError('Expected exact commit SHA')
 if not (ROOT/'.git').exists():raise ValueError('No Git repository:owner must create/commit it first;checker never initializes one')
 run=lambda *args:subprocess.run(['git','-C',str(ROOT),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30).stdout
 if Path(run('rev-parse','--show-toplevel').decode().strip()).resolve()!=ROOT.resolve():raise ValueError('Wrong Git repository')
 commit=run('rev-parse','--verify','HEAD').decode().strip()
 if not re.fullmatch('[0-9a-f]{40}',commit) or expected_commit is not None and commit!=expected_commit:raise ValueError('Wrong source commit')
 m=json.loads((ROOT/'LOCAL_SOURCE_MANIFEST.json').read_text('utf8'));expected=dict(m['files'])
 for name in ('LOCAL_SOURCE_MANIFEST.json','LOCAL_REPOSITORY_STATUS.md'):
  b=(ROOT/name).read_bytes();expected[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 child=subprocess.Popen(['git','-C',str(ROOT),'archive','--format=tar',commit],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 try:
  count=verify_archive(child.stdout,expected)
  if child.wait(timeout=30)!=0:raise ValueError('Local Git archive failed')
 finally:
  child.stdout.close()
  if child.poll() is None:child.kill()
  child.wait(timeout=5)
 return {'status':'PASS_COMMITTED_SOURCE_BYTES','commit':commit,'files':count,'source_manifest_sha256':expected['LOCAL_SOURCE_MANIFEST.json']['sha256'],'git_writes':0,'remote_operations':0}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--git-head',action='store_true');p.add_argument('--expected-commit');args=p.parse_args()
 if args.expected_commit and not args.git_head:p.error('--expected-commit requires --git-head')
 result=verify()
 if args.git_head:result['committed_source']=verify_git_head(args.expected_commit)
 print(json.dumps(result))
if __name__=='__main__':main()
