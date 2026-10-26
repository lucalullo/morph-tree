"""Thin installed Python interface; existing learner/admission/loader only.

This file is packaged as morphtree.api in the experimental wheel. An owned
execution budget is supplied internally by the existing runner, never inferred
from a constructor, a mode, a model file or an ordinary user call.
"""
import copy
import hashlib
import json
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from ._experimental.integrated import MorphTree as Integrated
from ._experimental.admission import AdmissionError, sha
from ._experimental.execution_v2 import require_session, ACTIVE, Session, PUBLIC_SCOPE

_USER_RUN_LOCK=threading.Lock()
PUBLIC_USER_ENABLED=True  # Enabled only by the explicitly versioned product build.
PUBLIC_USER_RUNTIME='2.0.0'


def _user_identity(package, q):
    """Self-check installed user distribution; NOT scientific qualification."""
    import base64,csv,hashlib,io,sys
    from ._experimental import admission as A
    package=Path(package).resolve()
    if q.get('usage')!='ORDINARY_USER_NOT_SCIENTIFIC':raise ValueError('Public-user identity domain required')
    manifest=A.read(package/'_experimental/manifest.json',expected_sha256=q['manifest_sha256'])
    if manifest['version']!=q['version'] or q['version']!=PUBLIC_USER_RUNTIME or A.digest(manifest)!=q['build_identity']:
        raise ValueError('User distribution manifest/version differs')
    files={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if files!=set(manifest['files'])|{'_experimental/manifest.json'}:raise ValueError('Unexpected/missing installed user component')
    # One byte snapshot must satisfy BOTH manifest and installed RECORD.
    # No cross-operation cache:every integrity boundary still reopens files.
    checked_content={}
    for n,h in manifest['files'].items():
        data=A.safe_path(package,n).read_bytes();raw_hash=hashlib.sha256(data).digest()
        if raw_hash.hex()!=h:raise ValueError('Altered installed user component')
        checked_content[n]=(raw_hash,len(data))
    data=(package/'_experimental/manifest.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=q['manifest_sha256']:raise ValueError('User manifest changed during verification')
    checked_content['_experimental/manifest.json']=(hashlib.sha256(data).digest(),len(data))
    infos=list(package.parent.glob('morphtree-*.dist-info'))
    if len(infos)!=1:raise ValueError('Unique distribution metadata required')
    info=infos[0];meta=info/'METADATA';record=info/'RECORD'
    if A.sha(meta)!=q['metadata_sha256'] or A.sha(record)!=q['record_sha256']:raise ValueError('Altered installed metadata/RECORD')
    if 'Name: morphtree\n' not in meta.read_text('utf8') or 'Version: '+q['version']+'\n' not in meta.read_text('utf8'):
        raise ValueError('User metadata name/version mismatch')
    rows={n:(h,size) for n,h,size in csv.reader(io.StringIO(record.read_text('utf8')))}
    for n in list(manifest['files'])+['_experimental/manifest.json']:
        h,size=rows.get('morphtree/'+n,('',0));raw_hash,actual_size=checked_content[n]
        expected='sha256='+base64.urlsafe_b64encode(raw_hash).decode().rstrip('=')
        if h!=expected or int(size)!=actual_size:raise ValueError('Installed user RECORD content differs')
    import morphtree
    if Path(morphtree.__file__).resolve()!=package/'__init__.py':raise ValueError('Shadowed public package')
    for n,m in tuple(sys.modules.items()):
        if n=='morphtree' or n.startswith('morphtree.'):
            f=getattr(m,'__file__',None)
            if f and not Path(f).resolve().is_relative_to(package):raise ValueError('Mixed user/source module origins')
    if q['environment']!=A.environment():raise ValueError('User execution environment changed')
    return {'runtime':'morphtree-user-installed','version':q['version'],'artifact_sha256':q['record_sha256'],
        'artifact_identity_kind':'INSTALLED_RECORD_NOT_ORIGINAL_WHEEL_OR_SCIENTIFIC_QUALIFICATION',
        'build_identity':q['build_identity'],'manifest_sha256':q['manifest_sha256'],'backend':'pure','environment':q['environment']}


def _user_qualification(package,root,source,protocol,seconds,user_limits):
    from ._experimental import admission as A
    manifest=package/'_experimental/manifest.json';m=A.read(manifest)
    infos=list(package.parent.glob('morphtree-*.dist-info'))
    if len(infos)!=1:raise ValueError('Unique installed user distribution required')
    return {'schema':'morph030-qualified-installed-v1','usage':'ORDINARY_USER_NOT_SCIENTIFIC',
        'version':m['version'],'backend':'pure','environment':A.environment(),'manifest_sha256':A.sha(manifest),
        'build_identity':A.digest(m),'metadata_sha256':A.sha(infos[0]/'METADATA'),'record_sha256':A.sha(infos[0]/'RECORD'),
        'input_root':str(root),'deadline_utc_epoch':time.time()+seconds,
        'limits':{'fits':128,'trials':user_limits['trials'],'operations':256,'seconds':seconds},'scalar_cap':2048,
        'sources':{A.digest(source):source},'components':[],'component_contracts':{},'protocol':protocol,
        'authorization':{'id':'USER-API:'+uuid.uuid4().hex,'operations':['inspect','load','save','export','fit','trial']},
        'optional_dependency':None}


def _user_source(source, expected):
    """No user-supplied path/role converts this scope into research authority."""
    from ._experimental import admission as A
    if source!=expected or set(source)!={'origin','exposure','sealed','rows','partitions','OOF'}:
        raise ValueError('Exact ordinary user source required')
    if source['exposure']!='USER_SUPPLIED_NOT_SCIENTIFIC' or source['sealed'] or source['OOF']:
        raise ValueError('Controlled/OOF/sealed source is not an ordinary input')
    ids=[r['id'] for r in source['rows']];parts=source['partitions']
    if not source['origin'].startswith('USER_IN_MEMORY:') or len(ids)!=len(set(ids)) or not ids:
        raise ValueError('User source identity/rows required')
    if set(parts)!={'TRAIN','SEARCH','GUARD_FINAL'} or parts['GUARD_FINAL']:
        raise ValueError('User scope never opens GUARD')
    union=parts['TRAIN']+parts['SEARCH']
    if not parts['TRAIN'] or not parts['SEARCH'] or len(union)!=len(set(union)) or set(union)!=set(ids):
        raise ValueError('Exact disjoint internal user split required')
    # Do not silently count duplicated observations as independent validation.
    rows=[A.digest({'X':r['X'],'y':r['y']}) for r in source['rows']]
    if len(rows)!=len(set(rows)):raise ValueError('Duplicate-row groups need a separately qualified splitting policy')
    return A.digest(source)


def _new_user_session(estimator, X, y, *, started=None):
    """Internal ordinary-user scope; never an owner/scientific capability.

    Input is supplied in memory, not looked up in a research workspace. The
    existing runtime/admission/store/executor are reused. No caller can use a
    failed active scientific context as a request to enter this branch.
    """
    from ._experimental import admission as A
    from ._experimental.target import detect_target, _split, _NamedRows
    from ._experimental.preprocessing import _m26_rows
    from ._experimental.producer import PIPE_CURRENT
    from ._experimental.runtime_io import configure
    from .config import HyperParams
    started=time.monotonic() if started is None else started
    scope=PUBLIC_SCOPE.get()
    if scope is not None:scope.before_input(estimator)
    if ACTIVE.get() is not None or A.CURRENT.get() is not None or PIPE_CURRENT.get() is not None:
        raise AdmissionError({'status':'REJECTED','reason':'RESEARCH_CONTEXT_NEVER_FALLS_THROUGH_TO_PUBLIC'})
    for obj in (X,y):
        if any(hasattr(obj,k) for k in ('_morphtree_source_role','_morphtree_scientific_source','_morphtree_sealed')):
            raise ValueError('Controlled-source objects require their explicit scientific context')
    estimator._validate()
    if estimator._research_growth:
        raise ValueError('Research operator overrides require a scientific execution context')
    # Dependency availability is checked before source materialization or fits.
    if estimator.tuning=='optuna':
        import importlib.util
        if importlib.util.find_spec('optuna') is None:
            raise ImportError('Final Optuna is a standard dependency; install optuna==5.0.0 or use_optuna=False skips only numerical tuning')
    rows,names=_m26_rows(X);task,targets,classes=detect_target(y,estimator.task)
    if len(rows)!=len(targets):raise ValueError('Aligned X/y required')
    if sum(4*len(v) if isinstance(v,str) else 8 for row in rows for v in row)>2*1024**2:
        raise ValueError('Current ordinary-user input exceeds the2MiB estimated payload safety cap')
    raw_y=list(targets) if task=='regression' else [classes[i] for i in targets]
    if scope is not None:scope.input_rows(rows,raw_y,names,task)
    seed=estimator.random_state
    if seed is None:
        import random
        seed=random.SystemRandom().randrange(2**31)
    ti,si=_split(targets,task,seed)
    if len(ti)>65536:raise ValueError('Current native component path supports at most65536 TRAIN rows')
    from ._experimental.preprocessing import TrainEncoder
    if TrainEncoder().fit([rows[i] for i in ti]).width_>128:
        raise ValueError('Current native component path supports at most128 encoded features')
    source={'origin':'USER_IN_MEMORY:'+A.digest({'X':rows,'y':raw_y,'names':names}),
        'exposure':'USER_SUPPLIED_NOT_SCIENTIFIC','sealed':False,
        'rows':[{'id':'ROW:'+str(i),'X':r,'y':raw_y[i]} for i,r in enumerate(rows)],
        'partitions':{'TRAIN':['ROW:'+str(i) for i in ti],'SEARCH':['ROW:'+str(i) for i in si],'GUARD_FINAL':[]},'OOF':[]}
    seconds=150. if estimator.time_budget=='auto' else float(estimator.time_budget)
    params=asdict(HyperParams(n_estimators=2,max_leaves=8,max_depth=4,max_bins=15,min_samples_leaf=2,row_fraction=1.))
    if estimator.capacity:params.update(estimator.capacity)
    # Original finite learning adapter supports <=8 stages/members and <=4
    # numerical trials. These are resource limits, not scientific grants.
    limits={'calls':128,'scalar_trees':8192,'trials':0 if estimator.tuning=='none' else 2 if estimator.n_trials=='auto' else estimator.n_trials,'artifact_bytes':32*1024**2}
    temporary=tempfile.TemporaryDirectory(prefix='morphtree-user-',dir=None if scope is None else scope.calls)
    root=Path(temporary.name).resolve();configure(root)
    if scope is not None:scope.retain_temporary(temporary,root)
    try:
        protocol={'schema':'morphtree-user-call-v1','purpose':'ORDINARY_USER_NOT_SCIENTIFIC',
            'id':uuid.uuid4().hex,'source_sha256':A.digest(source),'task':task,'seed':seed,'limits':limits,'seconds':seconds}
        pp=root/'USER-CALL.json';pp.write_text(json.dumps(protocol,sort_keys=True)+'\n',encoding='utf8')
        reference={'path':pp.name,'sha256':A.sha(pp)}
        q=A.user_qualification(Path(__file__).parent,root,source,reference,seconds,limits)
        preparation=time.monotonic()-started
        if preparation>=seconds:raise TimeoutError('User budget exhausted during input preparation')
        q['deadline_utc_epoch']-=preparation
        qp=root/'USER-ADMISSION.json';qp.write_text(json.dumps(q,sort_keys=True)+'\n',encoding='utf8')
        engine=A.AdmissionEngine(package=Path(__file__).parent,qualification_path=qp,
            qualification_sha256=A.sha(qp),workdir=root)
        plan={'schema':'comp002-v2-session-v1','mode':'PUBLIC_USER','source':source,
            'protocol':reference,'seed':seed,'params':params,'structural':estimator.morphing=='auto',
            'max_requests':128,'seconds':seconds,'user_limits':limits}
        session=Session(engine,root=root,plan=plan)
        session.started=started;session.engine.started=started
        session.event('USER_INPUT_PREPARED',seconds_operational=preparation,source_scope='USER_SUPPLIED_NOT_SCIENTIFIC')
        return temporary,session,_NamedRows(rows,names),raw_y,seed
    except BaseException:
        if scope is None:temporary.cleanup()
        raise


class TrainingNotEnabledError(RuntimeError):
    """This experimental run has no approved training execution budget."""


class MorphTree(Integrated):
    """COMP007 source: initial native exploration then bounded evolution.

    User code only supplies data and normal estimator options. The qualifying
    host owns execution authorization; this build does not grant unrestricted
    training. Adaptive may abstain; research requires an explicit operator and
    a separately admitted run. Neither choice implies scientific superiority.
    """
    def __init__(self, *, mode=None, operators=None, verbosity=None,
                 use_optuna=None, max_mutations=2, **kwargs):
        if type(max_mutations) is not int or not 0<=max_mutations<=8:
            raise ValueError('max_mutations must be an integer in [0,8]')
        if verbosity is not None:
            if type(verbosity) is not int or verbosity not in (0,1) or 'verbose' in kwargs:
                raise ValueError('Choose implemented verbosity 0/1 or legacy verbose, not both')
            kwargs['verbose']=verbosity
        if use_optuna is not None and type(use_optuna) is not bool:
            raise ValueError('use_optuna must be bool')
        if use_optuna is not None and 'tuning' in kwargs:
            raise ValueError('Choose use_optuna or legacy tuning, not both')
        # No silent optional-dependency skip: the existing explicit optuna path
        # checks the dependency and separate trial admission BEFORE any learning.
        if 'tuning' not in kwargs:
            kwargs['tuning']='none' if use_optuna is False else 'optuna'
        automatic=mode is None and not (set(kwargs)&{'families','morphing'})
        if automatic:mode='adaptive'
        if mode is not None:
            if mode not in ('baseline', 'adaptive', 'research'):
                raise ValueError('mode must be baseline, adaptive or research')
            if set(kwargs) & {'families', 'morphing'}:
                raise ValueError('Choose mode or legacy families/morphing, not both')
            kwargs.update(families='boost' if mode=='baseline' else 'auto',
                          morphing='none' if mode=='baseline' else 'auto')
        if operators is not None and (mode!='research' or tuple(operators)!=('growth_policy:depthwise',)):
            raise ValueError('Only explicit research growth_policy:depthwise is available')
        if mode=='research' and operators is None:
            raise ValueError('research requires explicit operators')
        super().__init__(**kwargs)
        if mode=='research': self._research_growth=True
        self._evolving_program=(not self._research_growth and self.families=='auto' and self.morphing=='auto')
        self._max_mutations=max_mutations

    def fit(self, X, y):
        if ACTIVE.get() is not None:
            # An invalid/expired/STOP scientific context MUST raise here, never
            # retry as ordinary user data. Original owner checks remain active.
            require_session()
            return super().fit(X,y)
        from ._experimental import admission as A
        from ._experimental.producer import PIPE_CURRENT
        from ._experimental.runtime_io import WORKDIR
        if A.CURRENT.get() is not None or PIPE_CURRENT.get() is not None:
            raise AdmissionError({'status':'REJECTED','reason':'ORPHANED_OPERATION_NOT_PUBLIC_DATA'})
        if not PUBLIC_USER_ENABLED:
            raise TrainingNotEnabledError('This installed version requires its explicitly approved execution context')
        if not _USER_RUN_LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent fit in this process is unsupported; use separate processes')
        work_token=WORKDIR.set(WORKDIR.get());old_temp=tempfile.tempdir
        workspace=None;session=None;old_seed=self.random_state
        before=dict(self.__dict__);closure_attempted=False;started=time.monotonic()
        try:
            workspace,session,xx,yy,seed=_new_user_session(self,X,y,started=started)
            self.random_state=seed
            with session.activate():result=super().fit(xx,yy)
            session._user_integrity()
            closure_attempted=True;session.close_user_run()
            self._public_user_run=True
            self._public_workspace=workspace
            self._public_resource_limits=copy.deepcopy(session.plan['user_limits'])
            self._cost=session.counters()
            return result
        except BaseException as exc:
            attempt=copy.deepcopy(getattr(self,'last_attempt_',{}))
            if attempt==before.get('last_attempt_',{}):
                attempt={'status':'FAIL','phase':'PUBLIC_ADMISSION','physical_fits':getattr(session,'_user_started',0),'trials':getattr(session,'_user_trials',0)}
            if attempt.get('status')=='PASS':attempt.update(status='FAIL',stop_reason='PUBLIC_FINALIZATION_FAILED_RESTORED_PRIOR_OBJECT')
            attempt.update(error=type(exc).__name__,message=str(exc))
            self.__dict__.clear();self.__dict__.update(before);self.last_attempt_=attempt
            if workspace is not None:self._failed_public_workspace=workspace
            raise
        finally:
            self.random_state=old_seed
            try:
                if session is not None and not closure_attempted:
                    closure_attempted=True;session.close_user_run()
            except BaseException as exc:
                self.last_attempt_.update(closure_status='UNKNOWN_FAILED',closure_error=type(exc).__name__)
                raise
            finally:
                tempfile.tempdir=old_temp;WORKDIR.reset(work_token)
                _USER_RUN_LOCK.release()

    @contextmanager
    def _public_output(self,path):
        """Explicit user output destination, never a scientific writer bypass."""
        from ._experimental import admission as A
        scope=PUBLIC_SCOPE.get()
        if scope is not None:scope.output(path)
        if ACTIVE.get() is not None or A.CURRENT.get() is not None or not (getattr(self,'_public_user_run',False) or getattr(self,'_input_sha256',None)):
            yield path;return
        from ._experimental.runtime_io import WORKDIR, configure
        target=Path(path).resolve();parent=target.parent
        if not parent.is_dir():raise ValueError('Output parent directory must exist')
        old=self._workdir;token=WORKDIR.set(WORKDIR.get());old_temp=tempfile.tempdir
        try:
            self._workdir=str(parent);configure(parent);yield target
        finally:
            self._workdir=old;tempfile.tempdir=old_temp;WORKDIR.reset(token)

    def save(self,path,*,variant='all'):
        with self._public_output(path) as p:return super().save(p,variant=variant)

    def save_recipe(self,path,*,variant='selected'):
        with self._public_output(path) as p:return super().save_recipe(p,variant=variant)

    def export_python(self,path,*,variant='selected'):
        with self._public_output(path) as p:return super().export_python(p,variant=variant)

    def export_algorithm(self,path):
        with self._public_output(path) as p:return super().export_algorithm(p)

    def set_params(self, **values):
        """Update implemented public controls without resetting evolution limits.

        The inherited legacy setter reconstructed the object without the
        public mutation limit/research operator. It also rejected canonical
        verbosity/use_optuna names. Serialization keeps its legacy constructor
        projection; executed evolution limits remain in the persisted trace.
        Validation is transactional: an invalid change preserves fitted state.
        """
        options=super().get_params()
        allowed=set(options)|{'mode','operators','verbosity','use_optuna','max_mutations'}
        if set(values)-allowed:
            raise ValueError('Unknown estimator parameter: '+', '.join(sorted(set(values)-allowed)))
        options['max_mutations']=self._max_mutations
        if self._research_growth:
            options.pop('families');options.pop('morphing')
            options.update(mode='research',operators=('growth_policy:depthwise',))
        # Remove an inherited legacy spelling only when it was not supplied
        # explicitly in the same update. Explicit conflicts remain errors.
        for canonical,legacy in [('verbosity','verbose'),('use_optuna','tuning')]:
            if canonical in values and legacy not in values:options.pop(legacy,None)
        if 'mode' in values:
            for legacy in ('families','morphing'):
                if legacy not in values:options.pop(legacy,None)
            if values['mode']!='research' and 'operators' not in values:
                options.pop('operators',None)
        options.update(values)
        replacement=type(self)(**options)
        self.__dict__=replacement.__dict__
        return self

    def export_standalone(self, path):
        """Export the existing dependency-free inference predictor, not training."""
        return self.export_python(path)

    @classmethod
    def load(cls, path, *, sha256=None, workdir=None):
        scope = PUBLIC_SCOPE.get()
        if scope is not None:
            scope.output(path)
            if workdir is not None:
                scope.output(workdir)
        source = Path(path).resolve()
        if not source.is_file() or not 0 < source.stat().st_size <= 32 * 1024 ** 2:
            raise ValueError('A bounded model file is required')
        if sha256 is None:
            with source.open('rb') as stream:
                raw = stream.read(32 * 1024 ** 2 + 1)
            if not 0 < len(raw) <= 32 * 1024 ** 2:
                raise ValueError('A bounded model file is required')
            expected = hashlib.sha256(raw).hexdigest()
            del raw
        else:
            expected = sha256
        from ._experimental.runtime_io import WORKDIR
        token = WORKDIR.set(WORKDIR.get())
        old_temp = tempfile.tempdir
        workspace = None
        try:
            if workdir is None:
                workspace = tempfile.TemporaryDirectory(prefix='morphtree-model-')
                workdir = workspace.name
            model = super().load(source, sha256=expected, workdir=workdir)
            if workspace is not None:
                model._loaded_workspace = workspace
            return model
        except BaseException:
            if workspace is not None:
                workspace.cleanup()
            raise
        finally:
            tempfile.tempdir = old_temp
            WORKDIR.reset(token)
