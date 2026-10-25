"""Portable experimental admission, zero learning budget; no checkout imports.

The caller pins a qualification dossier out of band. A self-supplied matching
name/version/hash is not trust. This cooperative guard is not an OS sandbox.
"""
import copy
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import time
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

OPERATIONS = frozenset(('inspect','load','infer','save','export','replay','optuna_import',
    'fit','trial','OOF','campaign','independent_evaluation','release_prepare'))
LEARNING = frozenset(('fit','trial','OOF','campaign'))
STATES = frozenset(('ADMITTED','REJECTED','BLOCKED','INVALID'))
CURRENT = ContextVar('morph030_exact_operation',default=None)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,limit=32*1024**2,*,expected_sha256=None):
    b=Path(path).read_bytes()
    if len(b)>limit:raise ValueError('Bounded input exceeded')
    if expected_sha256 is not None and hashlib.sha256(b).hexdigest()!=expected_sha256:raise ValueError('Changed bound input bytes')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    def bad(_):raise ValueError('Nonfinite JSON')
    v=json.loads(b.decode('utf8'),object_pairs_hook=pairs,parse_constant=bad)
    digest(v)
    return v


def safe_path(root,value):
    root=Path(root).resolve();p=Path(value)
    p=p if p.is_absolute() else root/p
    # Reject links/reparse redirections even when their final target is internal.
    for a in (p,*p.parents):
        if a.exists() and (a.is_symlink() or getattr(a.stat(),'st_file_attributes',0)&0x400):raise ValueError('Link/reparse path')
    p=p.resolve()
    if not p.is_relative_to(root):raise ValueError('Path outside explicit working directory')
    return p


def environment():
    return {'python':sys.version,'implementation':platform.python_implementation(),
        'platform':sys.platform,'executable_sha256':sha(sys.executable),'backend':'pure'}


def installed_identity(package,qualification):
    package=Path(package).resolve();q=qualification
    if q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC':
        from morphtree.api import _user_identity
        return _user_identity(package,q)
    if q.get('schema')!='morph030-qualified-installed-v1':raise ValueError('Unknown qualification schema')
    if q.get('version')!='1.0.0rc4' or q.get('backend')!='pure':raise ValueError('Unqualified version/backend')
    if environment()!=q['environment']:raise ValueError('Unqualified execution environment')
    manifest=package/'_experimental/manifest.json'
    m=read(manifest,expected_sha256=q['manifest_sha256'])
    if m['version']!=q['version'] or digest(m)!=q['build_identity']:raise ValueError('Incoherent build identity')
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=set(m['files'])|{'_experimental/manifest.json'}:raise ValueError('Missing/unexpected installed component')
    for name,h in m['files'].items():
        if sha(safe_path(package,name))!=h:raise ValueError('Changed installed component: '+name)
    infos=list(package.parent.glob('morphtree-*.dist-info'))
    if len(infos)!=1:raise ValueError('Unique experimental distribution metadata required')
    meta=infos[0]/'METADATA'
    metadata=meta.read_bytes()
    if hashlib.sha256(metadata).hexdigest()!=q['metadata_sha256']:raise ValueError('Changed metadata')
    text=metadata.decode('utf8')
    if 'Name: morphtree\n' not in text or 'Version: '+q['version']+'\n' not in text:raise ValueError('Metadata name/version mismatch')
    # local pip records the supplied wheel hash, not merely the package name.
    direct=read(infos[0]/'direct_url.json')
    if direct.get('archive_info',{}).get('hashes',{}).get('sha256')!=q['wheel_sha256']:raise ValueError('Installed wheel provenance mismatch')
    import morphtree
    if Path(morphtree.__file__).resolve()!=package/'__init__.py':raise ValueError('Shadowed runtime')
    for name,module in tuple(sys.modules.items()):
        if name=='morphtree' or name.startswith('morphtree.'):
            f=getattr(module,'__file__',None)
            if f and not Path(f).resolve().is_relative_to(package):raise ValueError('Mixed native module origins')
    return {'runtime':'morphtree-experimental-installed','version':q['version'],
        'artifact_sha256':q['wheel_sha256'],'build_identity':q['build_identity'],
        'manifest_sha256':q['manifest_sha256'],'backend':'pure','environment':environment()}


def validate_roles(source,expected):
    """Role is bound to actual row content/order and trusted source declaration.

    Origin/exposure is a provenance assertion pinned by the dossier, never
    inferred from a filename. These checks cannot discover dishonest originals.
    """
    if source!=expected:raise ValueError('Source/role/provenance differs from qualified declaration')
    if set(source)!={'origin','exposure','sealed','rows','partitions','OOF'}:raise ValueError('Exact source schema required')
    if source['sealed'] or source['exposure'] not in ('DEV_EXPOSED','SOFTWARE_FIXTURE'):raise ValueError('Sealed/independent source not admitted')
    rows=source['rows'];ids=[r['id'] for r in rows]
    if not ids or len(set(ids))!=len(ids):raise ValueError('Unique original row IDs required')
    # Aliased IDs cannot turn a row into two data roles.
    contents=[digest({'X':r['X'],'y':r['y']}) for r in rows]
    if len(set(contents))!=len(contents):raise ValueError('Ambiguous duplicate-row provenance requires another protocol')
    parts=source['partitions']
    if set(parts)!={'TRAIN','SEARCH','GUARD_FINAL'} or parts['GUARD_FINAL']:raise ValueError('No final GUARD access in this scope')
    t,s=parts['TRAIN'],parts['SEARCH']
    if not t or not s or len(set(t+s))!=len(t+s) or set(t+s)!=set(ids):raise ValueError('Exact disjoint TRAIN/SEARCH coverage required')
    folds=source['OOF']
    if folds:
        held=[]
        for f in folds:
            if set(f)!={'fit','held','recipe_sha256'} or set(f['fit'])&set(f['held']) or set(f['fit'])|set(f['held'])!=set(t):raise ValueError('OOF must use disjoint TRAIN-only folds')
            if len(set(f['fit']))!=len(f['fit']) or len(set(f['held']))!=len(f['held']) or not f['fit'] or not f['held']:raise ValueError('Nonempty unique OOF IDs')
            held+=f['held']
        if len(set(held))!=len(held) or set(held)!=set(t) or len({f['recipe_sha256'] for f in folds})!=1:raise ValueError('Fixed recipe/exact held coverage required')
    return digest(source)


def capacity(recipe,task):
    if recipe.get('schema')=='comp009-evolution-capacity-v1':
        from morphtree.config import AlgorithmGenome
        if set(recipe)!={'schema','initial','OOF','child_genome'} or task!='binary':
            raise ValueError('Exact evolution capacity schema/task')
        if recipe['OOF']!={'calls':8,'scalar_trees':8,'folds':2,'max_contexts':2} or recipe['child_genome']!=AlgorithmGenome(growth_policy='depthwise').to_dict():
            raise ValueError('Exact cumulative OOF/child capacity required')
        initial=recipe['initial']
        if set(initial)!={'candidates'} or len(initial['candidates'])!=2:
            raise ValueError('Two exact initial native forms')
        forms=initial['candidates']
        expected=[{'family':f,'members':1 if f=='boost' else 2,'stages':2 if f=='boost' else 1,
                   'classes':2,'genome_sha256':digest(AlgorithmGenome().to_dict())} for f in ('boost','bag')]
        if forms!=expected or capacity(initial,task)!=4:
            raise ValueError('Fixed initial native capacity')
        return 14
    if set(recipe)=={'candidates','research_genome'}:
        from morphtree.config import AlgorithmGenome
        if task!='binary' or recipe['research_genome']!=AlgorithmGenome(growth_policy='depthwise').to_dict():
            raise ValueError('Exact operative binary research genome required')
        base=capacity({'candidates':recipe['candidates']},task)
        return base+max(capacity(form,task) for form in recipe['candidates'])
    if set(recipe)=={'candidates'}:
        if type(recipe['candidates']) is not list or not 1<=len(recipe['candidates'])<=2:raise ValueError('Finite one/two native candidate path')
        families=[p['family'] for p in recipe['candidates']]
        if len(set(families))!=len(families):raise ValueError('No duplicate family budget alias')
        return sum(capacity(p,task) for p in recipe['candidates'])
    if set(recipe)!={'family','members','stages','classes','genome_sha256'} or recipe['family'] not in ('boost','bag'):raise ValueError('Bounded frozen native recipe required')
    for n in ('members','stages','classes'):
        if type(recipe[n]) is not int or recipe[n]<1:raise ValueError('Integer actual capacity required')
    if recipe['family']=='boost' and recipe['members']!=1 or recipe['family']=='bag' and recipe['stages']!=1:raise ValueError('Reset bag versus boost capacity mismatch')
    if task=='binary' and recipe['classes']!=2 or task=='regression' and recipe['classes']!=1 or task=='multiclass' and not 2<=recipe['classes']<=64:raise ValueError('Task/class capacity mismatch')
    return recipe['members']*recipe['stages']*(recipe['classes'] if task=='multiclass' else 1)


class AdmissionError(RuntimeError):
    def __init__(self,decision):
        self.decision=decision
        super().__init__(decision['status']+': '+decision['reason'])


class AdmissionEngine:
    """One finite, durable, cooperative zero-fit capability ledger.

    Admission authorizes exactly the supplied operation/request hash once. An
    unfinished INTENT is never silently replayed; physical execution is closed.
    """
    def __init__(self,*,package,qualification_path,qualification_sha256,workdir):
        self.q=read(qualification_path,expected_sha256=qualification_sha256);self.package=Path(package).resolve();self.workdir=Path(workdir).resolve()
        if not self.workdir.is_dir():raise ValueError('Explicit existing work directory required')
        self.identity=installed_identity(self.package,self.q);self._q_pin=digest(self.q)
        self.ledger=safe_path(self.workdir,'ADMISSION.jsonl');self.stop=safe_path(self.workdir,'STOP');self.lock=safe_path(self.workdir,'WRITER.lock')
        self.started=time.monotonic();self.deadline=self.q['deadline_utc_epoch']
        public=self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC'
        if not public and (self.q['limits']['fits']!=0 or self.q['limits']['trials']!=0):raise ValueError('No training authority in MORPH030')
        if public and (self.q['limits']['fits']!=128 or type(self.q['limits']['trials']) is not int or not 0<=self.q['limits']['trials']<=4):raise ValueError('Exact user caps required')
        if self.q['scalar_cap']!=(2048 if public else 32) or not 0<self.q['limits']['operations']<=(256 if public else 100) or not 0<self.q['limits']['seconds']<=(300 if public else 600):raise ValueError('Exact finite MORPH030 profile required')

    @contextmanager
    def _lease(self):
        fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        try:yield
        finally:os.close(fd);self.lock.unlink()

    def _events(self):
        if not self.ledger.exists():return []
        events=[];prev='0'*64
        for line in self.ledger.read_text('utf8').splitlines():
            v=json.loads(line);h=v.pop('sha256')
            if v['previous']!=prev or digest(v)!=h:raise ValueError('Corrupt/partial admission journal; reconcile, never replay')
            prev=h;events.append(v)
        return events

    def _event(self,kind,request_id,**values):
        old=self._events();v={'event':kind,'request_id':request_id,'previous':digest(old[-1]) if old else '0'*64,'utc':time.time(),'monotonic':time.monotonic(),**values};v['sha256']=digest(v)
        with self.ledger.open('a',encoding='utf8') as f:f.write(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

    def request(self,*,operation,task,recipe,source,role,protocol,authorization,components,requested_cost=None):
        r={'runtime':copy.deepcopy(self.identity),'operation':operation,'task':task,'recipe':copy.deepcopy(recipe),
            'effective_capacity':capacity(recipe,task),'source':copy.deepcopy(source),'role':role,
            'protocol':protocol,'authorization':authorization,'components':copy.deepcopy(components),
            'budget':copy.deepcopy(self.q['limits']),'deadline':self.deadline,'requested_cost':requested_cost or {'fits':0,'trials':0}}
        r['id']=digest(r);return r

    def decide(self,r):
        try:request_hash=digest(r)
        except (ValueError,TypeError,RecursionError):request_hash=None
        if type(r) is not dict or request_hash is None:return {'status':'INVALID','reason':'NONCANONICAL_REQUEST','request_sha256':request_hash,'operation':None,'permitted_operations':[],'physical_fits_authorized':0,'trials_authorized':0}
        def result(status,reason):return {'status':status,'reason':reason,'request_sha256':request_hash,'operation':r.get('operation'),'permitted_operations':[r['operation']] if status=='ADMITTED' else [],'physical_fits_authorized':r.get('requested_cost',{}).get('fits',0) if status=='ADMITTED' and self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC' else 0,'trials_authorized':r.get('requested_cost',{}).get('trials',0) if status=='ADMITTED' and self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC' else 0}
        try:
            if digest(self.q)!=self._q_pin:return result('INVALID','MUTATED_QUALIFICATION')
            if installed_identity(self.package,self.q)!=self.identity or r['runtime']!=self.identity:return result('REJECTED','RUNTIME_IDENTITY')
            if set(r)!={'id','runtime','operation','task','recipe','effective_capacity','source','role','protocol','authorization','components','budget','deadline','requested_cost'}|({'component_key'} if self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC' and r['operation'] in ('fit','trial') else set()):return result('INVALID','REQUEST_SCHEMA')
            if r['id']!=digest({k:v for k,v in r.items() if k!='id'}):return result('INVALID','REQUEST_ID')
            if r['operation'] not in OPERATIONS or r['task'] not in ('binary','multiclass','regression'):return result('INVALID','OPERATION_OR_TASK')
            if self.stop.exists() or any(e['event']=='STOP' for e in self._events()):return result('BLOCKED','STOP_PERMANENT')
            if time.time()>=self.deadline or time.monotonic()-self.started>=self.q['limits']['seconds'] or r['deadline']!=self.deadline:return result('BLOCKED','DEADLINE')
            if r['protocol']!=self.q['protocol'] or r['authorization']!=self.q['authorization']['id']:return result('REJECTED','PROTOCOL_OR_AUTHORIZATION')
            if digest(r['budget'])!=digest(self.q['limits']):return result('REJECTED','BUDGET_IDENTITY')
            c=capacity(r['recipe'],r['task'])
            if c!=r['effective_capacity']:return result('INVALID','MISREPORTED_CAPACITY')
            if c>self.q['scalar_cap']:return result('REJECTED','CAPACITY_ABOVE_32_QUALIFIED_PROFILE')
            key=digest(r['source'])
            if key not in self.q['sources']:return result('REJECTED','UNKNOWN_SOURCE_OR_RELABELLED_ROLE')
            _user_roles(r['source'],self.q['sources'][key]) if self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC' else validate_roles(r['source'],self.q['sources'][key])
            if r['role'] not in ('TRAIN','SEARCH','OOF','DEV'):return result('REJECTED','GUARD_OR_UNKNOWN_ROLE')
            if r['role']=='OOF' and not r['source']['OOF']:return result('REJECTED','OOF_PROVENANCE_NOT_SUPPLIED')
            if r['operation'] in LEARNING and r['role']!='TRAIN':return result('REJECTED','TRAINING_REQUIRES_TRAIN')
            if r['operation']=='replay' and r['role']!='SEARCH':return result('REJECTED','REPLAY_REQUIRES_SEARCH')
            if r['operation']=='independent_evaluation':return result('REJECTED','DEV_NOT_INDEPENDENT')
            for ref in r['components']:
                ev=self._events();completed={e['request_id'] for e in ev if e['event']=='COMPLETE'}
                produced=[e['artifact'] for e in ev if e['event']=='ARTIFACT' and e['request_id'] in completed]
                if ref not in self.q['components'] and ref not in produced:return result('REJECTED','UNQUALIFIED_COMPONENT')
                contracts=self.q['component_contracts']
                contract=contracts.get(ref['path'])
                if contract is None:
                    contract=next((e['contract'] for e in ev if e['event']=='ARTIFACT' and e['artifact']==ref and e['request_id'] in completed),None)
                if contract!={'task':r['task'],'recipe':r['recipe']}:return result('REJECTED','ACTUAL_COMPONENT_TASK_OR_CAPACITY')
                p=safe_path(self.q['input_root'],ref['path'])
                if p.stat().st_size!=ref['bytes'] or sha(p)!=ref['sha256']:return result('REJECTED','COMPONENT_HASH')
            ev=self._events();used={e['request_id'] for e in ev if e['event']=='INTENT'}
            if r['id'] in used:return result('BLOCKED','REQUEST_CONSUMED_OR_INCOMPLETE')
            if len(used)>=self.q['limits']['operations']:return result('BLOCKED','OPERATION_BUDGET_EXHAUSTED')
            if self.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC' and r['operation'] in ('fit','trial'):
                cost=r['requested_cost'];wanted={'fits':int(r['operation']=='fit'),'trials':int(r['operation']=='trial')}
                if cost!=wanted or any(type(v) is not int for v in cost.values()):return result('INVALID','USER_COST')
                if r['operation'] not in self.q['authorization']['operations']:return result('REJECTED','USER_OPERATION')
                key=r['component_key']
                if type(key) is not str or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):return result('INVALID','USER_COMPONENT_ID')
                for unit in ('fits','trials'):
                    if sum(e.get(unit,0) for e in ev if e['event']=='INTENT')+cost[unit]>self.q['limits'][unit]:return result('BLOCKED','USER_CUMULATIVE_BUDGET')
                return result('ADMITTED','EXACT_ORDINARY_USER_OPERATION_NOT_SCIENTIFIC')
            if digest(r['requested_cost'])!=digest({'fits':0,'trials':0}):return result('BLOCKED','LEARNING_BUDGET_ZERO')
            if r['operation'] in LEARNING:return result('BLOCKED','NEW_TRAINING_BOUNDARY_UNQUALIFIED_ZERO_BUDGET')
            if r['operation'] not in self.q['authorization']['operations']:return result('REJECTED','OPERATION_NOT_AUTHORIZED')
            return result('ADMITTED','EXACT_ZERO_FIT_OPERATION_ONLY')
        except (KeyError,ValueError,TypeError,OSError) as e:return result('INVALID',type(e).__name__+': '+str(e))

    @contextmanager
    def operation(self,r):
        with self._lease():
            d=self.decide(r)
            self._event('DECISION',r.get('id','INVALID'),decision=d)
            if d['status']!='ADMITTED':raise AdmissionError(d)
            self._event('INTENT',r['id'],request_sha256=digest(r),operation=r['operation'],fits=0,trials=0)
            token=CURRENT.set((self,copy.deepcopy(r)))
            try:
                yield d
                # Revalidate artifact/deadline/STOP after the callback without
                # treating our own already persisted intent as a replay.
                installed_identity(self.package,self.q)
                if self.stop.exists() or any(e['event']=='STOP' for e in self._events()) or time.time()>=self.deadline or time.monotonic()-self.started>=self.q['limits']['seconds']:raise RuntimeError('STOP/deadline before completion')
                self._event('COMPLETE',r['id'],fits=0,trials=0)
            except BaseException as e:
                self._event('INCOMPLETE',r['id'],error=type(e).__name__,fits=0,trials=0);raise
            finally:CURRENT.reset(token)

    @contextmanager
    def fold_operation(self,r):
        # The outer operation owns WRITER.lock across yield. Reuse that lease
        # only for this declared TRAIN-fold child, not arbitrary nested access.
        current=CURRENT.get()
        if current is None or current[0] is not self:
            raise ValueError('OOF child requires the active owned parent operation')
        outer=current[1]
        if outer['operation']!='inspect' or outer['role']!='TRAIN' or outer['recipe'].get('schema')!='comp009-evolution-capacity-v1' or r['operation']!='inspect' or r['role']!='OOF':
            raise ValueError('Exact versioned parent/OOF inspect relationship')
        if any(r['source'][k]!=outer['source'][k] for k in ('origin','rows','partitions','sealed','exposure')):
            raise ValueError('OOF source differs from actual parent TRAIN/SEARCH')
        d=self.decide(r);self._event('DECISION',r['id'],decision=d,parent_request=outer['id'])
        if d['status']!='ADMITTED':raise AdmissionError(d)
        self._event('INTENT',r['id'],request_sha256=digest(r),operation='inspect',parent_request=outer['id'],fits=0,trials=0)
        token=CURRENT.set((self,copy.deepcopy(r)))
        try:
            yield d
            installed_identity(self.package,self.q)
            if self.stop.exists() or time.time()>=self.deadline or time.monotonic()-self.started>=self.q['limits']['seconds']:
                raise RuntimeError('STOP/deadline before OOF completion')
            self._event('COMPLETE',r['id'],fits=0,trials=0)
        except BaseException as e:
            self._event('INCOMPLETE',r['id'],error=type(e).__name__,fits=0,trials=0);raise
        finally:CURRENT.reset(token)

    @contextmanager
    def user_learning(self,r):
        current=CURRENT.get()
        if self.q.get('usage')!='ORDINARY_USER_NOT_SCIENTIFIC' or current is None or current[0] is not self or current[1]['operation']!='inspect' or r['operation'] not in ('fit','trial'):
            raise PermissionError('Exact ordinary-user parent inspect required')
        d=self.decide(r);self._event('DECISION',r['id'],decision=d)
        if d['status']!='ADMITTED':raise AdmissionError(d)
        self._event('INTENT',r['id'],operation=r['operation'],fits=r['requested_cost']['fits'],trials=r['requested_cost']['trials'],domain='ORDINARY_USER_RESERVATION')
        try:
            yield d
            self._event('COMPLETE',r['id'],domain='ORDINARY_USER_RESERVATION')
        except BaseException as exc:
            self._event('INCOMPLETE',r['id'],error=type(exc).__name__,domain='ORDINARY_USER_RESERVATION');raise

    def record_output(self,path):
        current=CURRENT.get()
        if current is None or current[0] is not self or current[1]['operation'] not in ('save','export'):raise ValueError('Output requires current writer operation')
        p=safe_path(self.q['input_root'],path)
        ref={'path':p.relative_to(Path(self.q['input_root']).resolve()).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
        self._event('ARTIFACT',current[1]['id'],artifact=ref,contract={'task':current[1]['task'],'recipe':current[1]['recipe']})
        return ref

    def stop_permanently(self,reason):
        with self._lease():
            if not self.stop.exists():
                with self.stop.open('x',encoding='utf8') as f:f.write(str(reason));f.flush();os.fsync(f.fileno())
            self._event('STOP','CONTROLLER',reason=str(reason))

    def optional_optuna(self,r):
        if r['operation']!='optuna_import':raise ValueError('Import-only capability required')
        with self.operation(r):
            lock=self.q['optional_dependency']
            root=Path(lock['root']).resolve()
            for name,h in lock['files'].items():
                if sha(safe_path(root,name))!=h:raise ValueError('Changed optional dependency lock')
            import importlib.util
            spec=importlib.util.find_spec('optuna')
            if spec is None:raise ImportError('Optional Optuna absent')
            if Path(spec.origin).resolve()!=root/'optuna/__init__.py':raise ValueError('Unknown Optuna origin')
            module=importlib.import_module('optuna')
            if module.__version__!=lock['version']:raise ValueError('Incompatible Optuna version')
            return {'status':'ADMITTED_IMPORT_ONLY','version':module.__version__,'files_verified':len(lock['files']),'new_studies':0,'new_trials':0}


def user_qualification(package,root,source,protocol,seconds,user_limits):
    from morphtree.api import _user_qualification
    return _user_qualification(package,root,source,protocol,seconds,user_limits)
def _user_roles(source,expected):
    from morphtree.api import _user_source
    return _user_source(source,expected)
