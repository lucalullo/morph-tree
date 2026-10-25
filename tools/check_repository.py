"""Verify the exact local source snapshot without importing the learner."""
import argparse,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify():
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
 return {'status':'PASS_LOCAL_SOURCE_INTEGRITY','files':len(m['files']),'version':m['product']['version'],'release':'NO_GO','new_training':0}

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
