"""Version 2 research cache; preserves v1 source and fixes Pure backend spelling.

The caller must build identity from actual ordered inputs. A complete result is
immutable; uncertain jobs require reconciliation. OS leases prevent equal-key
concurrent computations. This is not a remote/untrusted model execution service.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib,json,math,os,re,time,uuid,sys
from pathlib import Path,PurePosixPath
from .runtime_io import artifact_path
from .processes import exclusive_lock,LockBusy
ROOT=None
CODE_ROOT=None

MAX_RECORD=2*1024**2
def canonical(v):
 text=json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
 if len(text.encode('utf-8'))>MAX_RECORD:raise ValueError('Bounded identity/receipt required')
 return text
def digest(v):return hashlib.sha256(canonical(v).encode('utf-8')).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _hash(h):
 if not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h):raise ValueError('SHA256 required')
 return h
def source_path(locator):
 if not isinstance(locator,str) or ':' in locator or '\\' in locator:raise ValueError('Project-relative source required')
 p=PurePosixPath(locator)
 if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Unsafe source locator')
 path=(CODE_ROOT.parent if p.parts[0]=='morphtree' else ROOT).joinpath(*p.parts)
 for a in (path,*path.parents):
  if a in (ROOT,CODE_ROOT.parent):break
  if a.is_symlink() or a.exists() and getattr(a.lstat(),'st_file_attributes',0)&0x400:raise ValueError('No source reparse traversal')
 if not path.is_file():raise ValueError('Missing source')
 return path
def _numeric(v):
 if v is None:return 'MISSING'
 if isinstance(v,bool) or not isinstance(v,(int,float)):raise ValueError('Explicit numeric encoded TRAIN values required')
 f=float(v)
 if math.isinf(f):raise ValueError('Infinite TRAIN values forbidden')
 return 'MISSING' if math.isnan(f) else f.hex()

@dataclass(frozen=True)
class FitRequest:
 text:str
 @property
 def key(self):return hashlib.sha256(self.text.encode('utf-8')).hexdigest()
 def value(self):return json.loads(self.text)

def learning_request(train,indices,*,protocol,split_sha256,preprocessing,task,family,structure,params,config,objective,seed,backend,dependencies,sources):
 """Freeze actual rows and all declared learner dependencies by value.

 Container weights/length, wall deadline and logical candidate name do not alter
 an atomic completed fit. Learning tree/sample budget does; interrupted work is
 never a cache hit. Protocol/split mismatches are intentionally conservative.
 """
 if type(train) is not dict or set(train)!={'role','ids','X','y'} or train['role']!='TRAIN':raise ValueError('TRAIN-only cache input')
 X,y,ids=train['X'],train['y'],train['ids']
 if not 1<=len(X)<=65536 or len(y)!=len(X) or len(ids)!=len(X) or len(set(ids))!=len(ids):raise ValueError('Aligned unique TRAIN rows required')
 if any(not isinstance(x,(str,int)) or isinstance(x,bool) for x in ids):raise ValueError('Opaque row identifiers required')
 width=len(X[0])
 if not 1<=width<=128 or any(len(row)!=width for row in X):raise ValueError('Bounded rectangular numeric TRAIN required')
 if task not in ('binary','regression','multiclass') or family not in ('boost','bag') or backend not in ('pure','numpy'):raise ValueError('Explicit native component context')
 if not indices or any(type(i) is not int or not 0<=i<len(X) for i in indices):raise ValueError('Ordered actual sample indices required')
 if type(seed) is not int or config.get('seed')!=seed or type(params.get('n_estimators')) is not int or params['n_estimators']<1:raise ValueError('Actual seed/tree budget required')
 if family=='bag' and (params['n_estimators']!=1 or params.get('learning_rate')!=1.):raise ValueError('Bag identity describes one reset member, not its container')
 if preprocessing.get('fit_role')!='TRAIN' or set(protocol)!={'path','sha256'} or not sources or not objective or not dependencies:raise ValueError('Full provenance required')
 from dataclasses import asdict
 from morphtree import HyperParams,AlgorithmGenome,MorphTreeConfig
 if set(params)!=set(asdict(HyperParams())) or set(config)!=set(asdict(MorphTreeConfig())) or set(structure)!=set(AlgorithmGenome().to_dict()):raise ValueError('Complete native params/config/genome required')
 HyperParams(**params);AlgorithmGenome(**structure);MorphTreeConfig(**config)
 if config['tuner']!='none' or config['enable_tuning'] or config['enable_morphing'] or config['refit_full'] or config['memory_enabled']:raise ValueError('Fixed state-free component fitting only')
 _hash(protocol['sha256']);_hash(split_sha256)
 values=[[ _numeric(v) for v in row] for row in X];targets=[_numeric(v) for v in y]
 # Hash the full actual input and effective ordered sample, not just array dimensions.
 h=hashlib.sha256()
 for row,label,ident in zip(values,targets,ids):h.update((canonical([row,label,ident])+'\n').encode('utf-8'))
 sampled=hashlib.sha256()
 for i in indices:sampled.update((canonical([values[i],targets[i],ids[i]])+'\n').encode('utf-8'))
 value={'schema':'native-component-learning-v1','protocol':protocol,'split_sha256':split_sha256,'role':'TRAIN','TRAIN_sha256':h.hexdigest(),
        'effective_sample_sha256':sampled.hexdigest(),'indices':list(indices),'rows':len(X),'features':width,'preprocessing':preprocessing,
        'task':task,'family':family,'structure':structure,'params':params,'config':config,'objective':objective,'seed':seed,'backend':backend,
        'dependencies':dependencies,'sources':sources,'learning_budget':{'trees':params['n_estimators'],'sample_rows':len(indices)}}
 required={'morphtree/'+p.name for p in CODE_ROOT.glob('*.py')}
 if not required:raise ValueError('Missing actual installed native module set')
 if not required<=set(sources):raise ValueError('Pin every consolidated learner module')
 for n,h in sources.items():_hash(h);source_path(n)
 return FitRequest(canonical(value))

def check_context(request):
 if type(request) is not FitRequest:raise ValueError('Canonical FitRequest required')
 v=request.value();p=v['protocol']
 if sha(source_path(p['path']))!=p['sha256']:raise ValueError('Changed protocol')
 for n,h in v['sources'].items():
  if sha(source_path(n))!=h:raise ValueError('Changed learner dependency')
 return v

def _write(path,value):
 path=artifact_path(path);text=canonical(value)+'\n'
 with path.open('x',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
def _read(path):
 path=artifact_path(path)
 if path.stat().st_size>MAX_RECORD:raise ValueError('Oversize cache record')
 return json.loads(path.read_text('utf-8'))

def check_model(path,value):
 from .execution_v2 import validate_model
 validate_model(path,value)

class PendingComponent(RuntimeError):pass
class ComputeBudgetExhausted(RuntimeError):pass

class ComponentStore:
 def __init__(self,folder,*,max_compute_jobs=0,seconds=180):
  self.root=artifact_path(folder);self.root.mkdir(exist_ok=True)
  if type(max_compute_jobs) is not int or max_compute_jobs<0 or not 0<seconds<=600:raise ValueError('Finite cache policy required')
  with exclusive_lock(artifact_path(self.root/'metadata-lock')):
   p=artifact_path(self.root/'POLICY.json')
   if not p.exists():_write(p,{'schema':'native-component-store-v1','max_compute_jobs':max_compute_jobs,'seconds':seconds,'expires_unix':time.time()+seconds})
   self.policy=_read(p)
   if self.policy['max_compute_jobs']!=max_compute_jobs or self.policy['seconds']!=seconds:raise ValueError('Cache policy cannot reset budget/deadline')
  self.keys=artifact_path(self.root/'components');self.keys.mkdir(exist_ok=True)
 def _active(self,*,reuse_only=False):
  if (self.root/'STOP').exists() or not reuse_only and time.time()>=self.policy['expires_unix']:raise ComputeBudgetExhausted('Cache computation expiry/STOP; no self-renewal')
 def event(self,kind,**fields):
  # Callers never acquire a component lock while holding this short journal lease.
  for attempt in range(20):
   try:
    with exclusive_lock(artifact_path(self.root/'journal-lock')):
     p=artifact_path(self.root/'EVENTS.jsonl')
     with p.open('a',encoding='utf-8') as f:f.write(canonical({'event':kind,'UTC_unix':time.time(),**fields})+'\n');f.flush();os.fsync(f.fileno())
    return
   except LockBusy:
    if attempt==19:raise
    time.sleep(min(.01*(attempt+1),.1))
 def _result(self,request,folder):
  row=_read(folder/'COMPLETE.json')
  if row['key']!=request.key or row['request']!=request.value():raise ValueError('Cache identity mismatch')
  ref=row['artifact'];path=artifact_path(ROOT/ref['path'])
  if path.stat().st_size!=ref['bytes'] or sha(path)!=ref['sha256']:raise ValueError('Changed completed component')
  for proof in row['proof']:
   if sha(source_path(proof['path']))!=proof['sha256']:raise ValueError('Changed completion provenance')
  return row
 def import_completed(self,request,model,*,expected_sha256,proof):
  """Explicit verified historical admission; this performs no learning job."""
  self._active();v=check_context(request);model=artifact_path(model);_hash(expected_sha256)
  if not proof or sha(model)!=expected_sha256 or model.stat().st_size>16*1024**2:raise ValueError('Verified bounded artifact required')
  for row in proof:
   if sha(source_path(row['path']))!=_hash(row['sha256']):raise ValueError('Bad completion evidence')
  check_model(model,v)
  folder=artifact_path(self.keys/request.key)
  with exclusive_lock(folder):
   if (folder/'COMPLETE.json').exists():return self._result(request,folder)
   if (folder/'INTENT.json').exists():raise PendingComponent('Reconcile interrupted learning before import')
   record={'key':request.key,'request':v,'artifact':{'path':model.relative_to(ROOT).as_posix(),'sha256':expected_sha256,'bytes':model.stat().st_size},'proof':proof,'origin':'VERIFIED_HISTORICAL_IMPORT','compute_jobs':0}
   _write(folder/'COMPLETE.json',record);self.event('IMPORTED_COMPLETED',key=request.key)
   return record
 @contextmanager
 def claim(self,request,*,logical_id):
  self._active(reuse_only=True);check_context(request)
  if not isinstance(logical_id,str) or not 1<=len(logical_id)<=200:raise ValueError('Bounded logical request ID')
  self.event('LOGICAL_REQUEST',key=request.key,logical_id=logical_id)
  folder=artifact_path(self.keys/request.key)
  try:
   with exclusive_lock(folder):
    if (folder/'COMPLETE.json').exists():
     result=self._result(request,folder);self.event('CACHE_HIT',key=request.key,logical_id=logical_id);yield Lease(self,request,folder,result);return
    self._active()
    if (folder/'INTENT.json').exists():raise PendingComponent('Uncommitted attempt must not replay')
    with exclusive_lock(artifact_path(self.root/'metadata-lock')):
     used=sum(1 for _ in self.keys.glob('*/INTENT.json'))
     if used>=self.policy['max_compute_jobs']:raise ComputeBudgetExhausted('No compute job budget')
     _write(folder/'INTENT.json',{'key':request.key,'request':request.value(),'owner_pid':os.getpid(),'logical_id':logical_id})
    self.event('COMPUTE_JOB_RESERVED',key=request.key,logical_id=logical_id)
    lease=Lease(self,request,folder,None);yield lease
    if lease.result is None:raise PendingComponent('Job incomplete; intent retained')
  except LockBusy:
   self.event('IN_FLIGHT_REFUSED',key=request.key,logical_id=logical_id);raise
 def counts(self):
  rows=[json.loads(x) for x in (self.root/'EVENTS.jsonl').read_text('utf-8').splitlines()] if (self.root/'EVENTS.jsonl').exists() else []
  return {n:sum(x['event']==n for x in rows) for n in ('LOGICAL_REQUEST','CACHE_HIT','COMPUTE_JOB_RESERVED','IMPORTED_COMPLETED','IN_FLIGHT_REFUSED','NATIVE_FIT_STARTED','NATIVE_FIT_COMPLETED')}

class Lease:
 def __init__(self,store,request,folder,result):self.store,self.request,self.folder,self.result=store,request,folder,result
 @property
 def hit(self):return self.result is not None
 def commit(self,model,*,expected_sha256,proof):
  if self.result is not None:raise ValueError('Completed lease is immutable')
  self.store._active();check_context(self.request);path=artifact_path(model)
  if not path.is_relative_to(self.folder) or path.stat().st_size>16*1024**2 or sha(path)!=_hash(expected_sha256) or not proof:raise ValueError('Owned complete artifact/provenance required')
  for ref in proof:
   if sha(source_path(ref['path']))!=_hash(ref['sha256']):raise ValueError('Changed provenance')
  check_model(path,self.request.value())
  _write(self.folder/'COMPLETE.json',{'key':self.request.key,'request':self.request.value(),'artifact':{'path':path.relative_to(ROOT).as_posix(),'sha256':expected_sha256,'bytes':path.stat().st_size},'proof':proof,'origin':'COMPUTED_UNDER_EXCLUSIVE_LEASE','compute_jobs':1})
  self.result=self.store._result(self.request,self.folder);self.store.event('COMPUTE_JOB_COMPLETED',key=self.request.key)

def native_binary_component(store,train,indices,*,logical_id,context):
 """One component adapter. A hit returns before estimator construction/fit.

 Current qualification covers verified hit/no-budget/role paths only. A future
 fit-capable campaign must independently qualify the miss branch and environment.
 """
 request=learning_request(train,indices,**context);v=request.value()
 if v['task']!='binary' or v['objective']!='binary_logistic_Newton':raise ValueError('This adapter qualifies binary fixed-objective components only')
 with store.claim(request,logical_id=logical_id) as lease:
  reused=lease.hit
  if not lease.hit:
   if not v['dependencies'].get('qualified_fit_environment_sha256'):raise ValueError('Fresh qualified fit environment required')
   from morphtree import MorphTreeClassifier,AlgorithmGenome,HyperParams,MorphTreeConfig
   model=MorphTreeClassifier(genome=AlgorithmGenome(**v['structure']),params=HyperParams(**v['params']),config=MorphTreeConfig(**v['config']),backend=v['backend'])
   store.event('NATIVE_FIT_STARTED',key=request.key)
   model.fit([train['X'][i] for i in indices],[train['y'][i] for i in indices],cancel=lambda:(store._active() or False))
   path=artifact_path(lease.folder/'model.mtree');model.save_model(path)
   lease.commit(path,expected_sha256=sha(path),proof=[v['protocol']]);store.event('NATIVE_FIT_COMPLETED',key=request.key)
  from morphtree import load_model
  return load_model(ROOT/lease.result['artifact']['path'],backend='pure'),{'key':request.key,'artifact':lease.result['artifact'],'reused':reused}

def configure(root,package):
 global ROOT,CODE_ROOT
 ROOT=Path(root).resolve();CODE_ROOT=Path(package).resolve()
 if not ROOT.is_dir() or not CODE_ROOT.is_dir():raise ValueError('Explicit existing roots')
def native_sources():
 return {'morphtree/'+p.relative_to(CODE_ROOT).as_posix():sha(p) for p in CODE_ROOT.rglob('*.py')}

class MultitaskStore(ComponentStore):

    def events(self):
        p = self.root / 'EVENTS.jsonl'
        if not p.exists():
            return []
        raw = p.read_bytes()
        if raw and (not raw.endswith(b'\n')):
            raise ValueError('Partial journal tail requires reconciliation')
        rows = []
        previous = '0' * 64
        for i, line in enumerate(raw.splitlines()):
            row = json.loads(line)
            body = {k: v for k, v in row.items() if k != 'sha256'}
            if row.get('sequence') != i + 1 or row.get('previous') != previous or row.get('sha256') != digest(body):
                raise ValueError('Corrupt execution journal')
            previous = row['sha256']
            rows.append(row)
        return rows

    def event(self, kind, **fields):
        with exclusive_lock(artifact_path(self.root / 'journal-lock')):
            rows = self.events()
            body = {'sequence': len(rows) + 1, 'previous': rows[-1]['sha256'] if rows else '0' * 64, 'event': kind, 'UTC_unix': time.time(), **fields}
            row = {**body, 'sha256': digest(body)}
            with (self.root / 'EVENTS.jsonl').open('a', encoding='utf8') as f:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())

    def _result(self, request, folder):
        row = super()._result(request, folder)
        validate_native(artifact_path(row['artifact']['path']), request)
        return row

def validate_native(path,request):
 from .execution_v2 import validate_model
 return validate_model(path,request.value())
