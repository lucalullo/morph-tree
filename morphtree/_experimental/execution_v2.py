"""Thin installed v2 adapter: existing admission/store/native executor only.

Software fixture operations reserve cache intents but never call native fit.
LIVE is intentionally closed until the complete finite worker/owner activation
profile is separately qualified; no API call, cache or old permit opens it.
"""
import copy
import json
import time
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from morphtree.config import AlgorithmGenome, HyperParams
from morphtree.persistence import read_json, LoadLimits
from .admission import AdmissionError, digest, sha, installed_identity
from .runtime_io import configure, artifact_path, WORKDIR, reference
from .target import _NamedRows, _split, detect_target
from .preprocessing import _m26_rows
from .training import NativeExecutor, NATIVE
from . import component_cache_v2 as cache

ACTIVE=ContextVar('comp002-owned-v2-session',default=None)
PUBLIC_SCOPE=ContextVar('product004-owned-public-qualification',default=None)


class PreFitBoundary(RuntimeError):
    pass


class SearchBudgetExhausted(TimeoutError):
    """Search cutoff inside the unchanged hard call budget;not a new permit."""
    pass


def require_session():
    session=ACTIVE.get()
    if type(session) is not Session:
        from .producer import Session as MultiphaseSession, PIPE_CURRENT
        if type(session) is not MultiphaseSession or PIPE_CURRENT.get()!=session.policy_pin:
            raise AdmissionError({'status':'BLOCKED','reason':'EXACT_INSTALLED_V2_SESSION_REQUIRED'})
    session.check()
    return session


def validate_model(path, value):
    """M026B0 task checks, with exact original producer seed before safe load.

    Seed stays in the request/provenance, not restored from arbitrary report.
    Loaded normalized reports are not mistaken for fresh producer witnesses.
    """
    task=value['task']; data=read_json(path, LoadLimits(max_bytes=16*1024**2))
    expected={'binary':'MorphTreeModel','regression':'MorphTreeRegressionModel','multiclass':'MorphTreeMulticlassModel'}[task]
    if task!='binary' and value['structure']!=AlgorithmGenome().to_dict():
        from morphtree.config import TASK_GROWTH_FORMATS,validate_task_genome
        validate_task_genome(task,AlgorithmGenome(**value['structure']))
        expected=TASK_GROWTH_FORMATS[task]
    if data.get('format')!=expected:
        raise ValueError('Native task mismatch')
    params=data['algorithm']['parameters'] if task=='binary' else data['parameters']
    if params!=value['params'] or data['n_features']!=value['features']:
        raise ValueError('Native actual parameters/features differ')
    if task=='binary' and data['algorithm']['genome']!=value['structure']:
        raise ValueError('Native genome differs')
    if task!='binary' and data.get('genome',AlgorithmGenome().to_dict())!=value['structure']:
        raise ValueError('Native nonbinary payload and actual requested genome differ')
    if task=='multiclass' and (data['strategy']!='ovr' or data['classes']!=value['preprocessing']['encoded_classes']):
        raise ValueError('OvR columns differ')
    # Exact producer payload is persisted before the safe loader projects report.
    if data['report'].get('backend')!='python-stdlib' or data['report'].get('config',{}).get('seed')!=value['seed']:
        raise ValueError('Missing/mismatched original producer seed/backend witness')
    model=NATIVE[task]._from_payload(data, limits=LoadLimits(max_bytes=16*1024**2), backend='pure')
    k=len(value['preprocessing']['encoded_classes']) if task=='multiclass' else 1
    if len(model.model_.trees)!=value['params']['n_estimators']*k:
        raise ValueError('Actual scalar trees differ')
    return model


class Session:
    """Adapter to the qualified finite primitives, not a second scheduler.

    One explicit AdmissionEngine and immutable source/plan. Canonical per-key
    leases/INTENT/COMPLETE/STOP come from ComponentStore. No implicit retraining.
    """
    def __init__(self, engine, *, root, plan, fixture=None):
        if ACTIVE.get() is not None:raise RuntimeError('No nested session construction')
        configure(root); self.root=artifact_path(root); self.engine=engine
        if self.root!=engine.workdir or self.root!=Path(engine.q['input_root']).resolve():
            raise ValueError('Exact admitted input/output scope required')
        self.plan=copy.deepcopy(plan); self.pin=digest(plan)
        public=plan.get('mode')=='PUBLIC_USER'
        required={'schema','mode','source','protocol','seed','params','structural','max_requests','seconds'}|({'user_limits'} if public else set())
        if set(plan)!=required or plan['schema']!='comp002-v2-session-v1':
            raise ValueError('Exact finite versioned profile required')
        if plan['protocol']!=engine.q['protocol']:
            raise ValueError('Session/AdmissionEngine protocol mismatch')
        self.mode=plan['mode']; self.structural=plan['structural']
        if self.mode not in ('FIXTURE','PREFIT','LIVE','PUBLIC_USER') or type(self.structural) is not bool:
            raise ValueError('Explicit mode/structural opt-in required')
        if self.mode=='LIVE':
            raise AdmissionError({'status':'BLOCKED','reason':'COMP002_LIVE_WORKER_OWNER_PROFILE_NOT_YET_QUALIFIED'})
        if self.mode=='FIXTURE' and not callable(fixture) or self.mode!='FIXTURE' and fixture is not None:
            raise ValueError('Explicit fixture provider only in software mode')
        if public and engine.q.get('usage')!='ORDINARY_USER_NOT_SCIENTIFIC':
            raise ValueError('Public Session cannot use a scientific qualification')
        if not public and engine.q.get('usage')=='ORDINARY_USER_NOT_SCIENTIFIC':
            raise ValueError('User qualification cannot enter scientific/fixture mode')
        if type(plan['max_requests']) is not int or not 1<=plan['max_requests']<=(128 if public else 32) or not 0<plan['seconds']<=(300 if public else 180):
            raise ValueError('Frozen software scope at most32 requests/180s')
        HyperParams(**plan['params'])
        if plan['params']['n_estimators']>8 or type(plan['seed']) is not int:
            raise ValueError('Existing finite adapter bound')
        self.fixture=fixture; self.started=time.monotonic(); self.prepared=None
        self.physical=0; self.prefit=0; self.fixture_calls=0
        self.component=None; self.logical=0; self.root.mkdir(exist_ok=True)
        self._oof_capture=None; self._diagnostic_proofs={}
        if public:
            limits=plan['user_limits']
            if set(limits)!={'calls','scalar_trees','trials','artifact_bytes'} or limits['calls']!=128 or limits['scalar_trees']!=8192 or type(limits['trials']) is not int or not 0<=limits['trials']<=4 or limits['artifact_bytes']!=32*1024**2:
                raise ValueError('Exact bounded ordinary-user resource profile required')
            if engine.q['limits']['fits']!=limits['calls'] or engine.q['limits']['trials']!=limits['trials']:
                raise ValueError('User admission and execution budgets must match')
            self._user_limits=copy.deepcopy(limits);self.phase='FAMILY';self.frozen=None;self._user_trials=0
            self._user_started=0;self._user_completed=0;self._user_trees=0;self._closed=False
            self._search_deadline=None;self._search_closed=False
            self._component_costs=[]
        policy=artifact_path('SESSION.json')
        if policy.exists(): raise ValueError('New software session; never renew/replay a partial run')
        cache.configure(self.root,engine.package)
        cache._write(policy,{'plan':self.plan,'runtime':engine.identity,'deadline':engine.deadline})
        self.policy_sha=sha(policy)
        self.store=cache.MultitaskStore(artifact_path('store'),max_compute_jobs=plan['max_requests'],seconds=plan['seconds'])
        self.public_scope=PUBLIC_SCOPE.get() if public else None
        if self.public_scope is not None:self.public_scope.bind_session(self)
        self.check()

    def check(self):
        if self.mode=='PUBLIC_USER':
            if self.public_scope is not None:
                if PUBLIC_SCOPE.get() is not self.public_scope:raise PermissionError('Lost public campaign binding')
                self.public_scope.check()
            # Hot cooperative cancellation checks do not hash the complete
            # dataset/runtime on every split. Full integrity remains mandatory
            # at activation, component entry and before component commitment.
            if self._closed or self.engine.stop.exists() or (self.root/'STOP').exists() or (self.store.root/'STOP').exists():
                raise RuntimeError('STOP permanent for this user call')
            if time.time()>=self.engine.deadline or time.monotonic()-self.started>=self.plan['seconds']:
                raise TimeoutError('Finite ordinary-user deadline')
            if getattr(self,'_component_deadline',None) is not None and time.monotonic()>=self._component_deadline:
                raise TimeoutError('Ordinary-user cooperative component deadline')
            if (not self._search_closed and self._search_deadline is not None
                    and time.monotonic()>=self._search_deadline):
                raise SearchBudgetExhausted('Search window ended;hard call budget is unchanged')
            return
        configure(self.root);cache.configure(self.root,self.engine.package)
        if digest(self.plan)!=self.pin or sha(self.root/'SESSION.json')!=self.policy_sha:
            raise ValueError('Changed immutable session')
        if digest(self.engine.q)!=self.engine._q_pin:
            raise ValueError('Mutated qualification dossier')
        if installed_identity(self.engine.package,self.engine.q)!=self.engine.identity:
            raise ValueError('Changed installed runtime')
        if self.engine.stop.exists() or (self.root/'STOP').exists() or (self.store.root/'STOP').exists():
            raise RuntimeError('STOP permanent including cache hits')
        if time.time()>=self.engine.deadline or time.monotonic()-self.started>=self.plan['seconds']:
            raise TimeoutError('Original finite deadline')

    def begin_search(self):
        if self.mode!='PUBLIC_USER':return
        self.check()
        if self._search_deadline is not None or self._search_closed:raise PermissionError('Search cannot renew')
        reserve=min(10.,self.plan['seconds']*.1)
        self._search_deadline=self.started+self.plan['seconds']-reserve
        self.event('USER_SEARCH_WINDOW',finalization_reserve_seconds=reserve,hard_seconds=self.plan['seconds'])

    def finish_search(self,reason):
        if self.mode!='PUBLIC_USER' or ACTIVE.get() is not self:
            raise PermissionError('Only the active ordinary user call may close search')
        # Irreversible closing can suppress only the SOFT search cutoff. Hard
        # deadline,STOP,campaign binding and component deadline remain checked.
        self._search_closed=True
        self.check()
        self.event('USER_SEARCH_CLOSED',reason=reason,new_learning_allowed=False)

    def _learning_open(self):
        self.check()
        if self.mode=='PUBLIC_USER' and self._search_closed:
            raise PermissionError('Search closed;only retained-state finalization remains')

    def admit_batch(self,task,family,params,*,folds=1,kind='FAMILY'):
        """Conservative preflight,not a reservation or a timing guarantee.

        Actual physical/global reservations remain at the native boundary.
        A batch that cannot fit the remaining quota is never partially started.
        Observed inclusive component costs only screen likely overruns;hard
        cancellation still applies and uncommitted children are never adopted.
        """
        if self.mode!='PUBLIC_USER':return
        self._learning_open()
        if task!=self.prepared['task'] or family not in ('boost','bag') or type(folds) is not int or not 1<=folds<=2:
            raise ValueError('Exact task/family and at most two TRAIN folds required')
        p=HyperParams(**params);calls=folds*(1 if family=='boost' else p.n_estimators)
        scalar=folds*p.n_estimators*(len(self.prepared['classes']) if task=='multiclass' else 1)
        reason=None
        if self._user_started+calls>self._user_limits['calls'] or self._user_trees+scalar>self._user_limits['scalar_trees']:
            reason='WHOLE_BATCH_EXCEEDS_REMAINING_CALL_QUOTA'
        if self.public_scope is not None:
            used=self.public_scope.used();profile=self.public_scope.policy['profile']
            if used['calls']+calls>profile['calls'] or used['scalar_trees']+scalar>profile['scalar_trees']:
                reason='WHOLE_BATCH_EXCEEDS_REMAINING_GLOBAL_QUOTA'
        observed=max(self._component_costs,default=0.)
        estimate=1.5*observed*calls if observed else None
        remaining=None if self._search_deadline is None else self._search_deadline-time.monotonic()
        if estimate is not None and remaining is not None and estimate>=remaining:
            reason='OBSERVED_COST_SCREEN_NO_ROOM_FOR_WHOLE_BATCH'
        self.event('USER_BATCH_PREFLIGHT',batch_kind=kind,task=task,family=family,calls=calls,
            scalar_trees=scalar,estimated_seconds=estimate,remaining_search_seconds=remaining,
            timing_scope='OPERATIONAL_HEURISTIC_NOT_GUARANTEED_OR_BENCHMARK',
            decision='STOP_BEFORE_BATCH' if reason else 'ALLOW_WITH_EXISTING_PER_COMPONENT_GUARDS',reason=reason)
        if reason:raise SearchBudgetExhausted(reason)

    @contextmanager
    def activate(self):
        if self.mode=='PUBLIC_USER':self._user_integrity()
        self.check()
        if ACTIVE.get() is not None: raise RuntimeError('No nested concurrent session')
        with cache.exclusive_lock(artifact_path('session-writer')):
            token=ACTIVE.set(self)
            try: yield self
            finally: ACTIVE.reset(token)

    def event(self, kind, **values):
        if self.mode=='PUBLIC_USER':
            op=values.get('value',{}).get('operation')
            if kind=='ORCHESTRATION' and op=='evolution_intent':
                if self.frozen is not None:raise ValueError('No mutation after public freeze')
                self.phase='STRUCTURE'
            if kind=='ORCHESTRATION' and op=='structure_frozen' and 'packet' in values['value']:
                if self.frozen is not None and self.frozen!=values['value']['packet']:
                    raise ValueError('Frozen user program changed')
                self.frozen=copy.deepcopy(values['value']['packet']);self.phase='NUMERIC'
            self.store.event(kind,phase=self.phase,**values);return
        self.store.event(kind, **values)

    def counters(self):
        p=self.store.root/'EVENTS.jsonl'
        rows=[json.loads(line) for line in p.read_text('utf8').splitlines()] if p.exists() else []
        result={'physical_fits':self.physical,'fixture_operations':self.fixture_calls,
                'prefit_entries':self.prefit,'trials':0,
                'OOF':sum(r['event']=='NATIVE_FIT_COMPLETED' and r.get('phase')=='OOF' for r in rows),
                'OOF_count_semantics':'COMPLETED_PHYSICAL_NATIVE_FIT_COMPONENTS_NOT_FIXTURE_OR_CACHE',
                'OOF_fixture_components':sum(r.get('components',0) for r in rows
                    if r['event']=='OOF_CONTEXT_PRODUCED' and r.get('physical_fits_scope')=='FIXTURE'),
                'calibrators':0,
                'store':self.store.counts(),'elapsed_operational_seconds':time.monotonic()-self.started,
                'CPU':'UNKNOWN','peak_RSS':'UNKNOWN','performance_qualified':False}
        if self.mode=='PUBLIC_USER':
            created=[r for r in rows if r['event']=='NUMERIC_TRIAL_ASKED']
            result.update(physical_fits_started=self._user_started,physical_fits_completed=self._user_completed,
                scalar_trees_started=self._user_trees,Optuna_trials=sum(r['real'] for r in created),trials=sum(r['real'] for r in created),
                trial_reservations=self._user_trials,simulated_trials=sum(not r['real'] for r in created),
                scope='ORDINARY_USER_OPERATIONAL_NOT_SCIENTIFIC',closed=self._closed)
        return result

    def evolution_context(self, selected, program):
        """Use only the exact Session's admitted fixed-parent fold operation.

        Base software sessions stay fixture-only. A versioned owned worker may
        implement oof_operation, with separate owner/role/phase/cumulative caps.
        Constructing a context never creates that authorization.
        """
        self.check()
        # A repeated unchanged parent may reuse a revalidated context; it does
        # not consume another inspect token or replay physical fitting.
        for proof in getattr(self,'_diagnostic_proofs',{}).values():
            if proof['parent']==selected and proof['context']['program_sha256']==digest(program):
                if self.verify_evolution_context(proof['context'],selected):
                    self.event('OOF_CONTEXT_REUSED',context=digest(proof['context']),physical_fits=0)
                    return copy.deepcopy(proof['context'])
        with self.oof_operation(selected,program) as evidence:
            if evidence is None:return None
            return self._produce_oof_context(selected,program,evidence)

    @contextmanager
    def oof_operation(self, selected, program):
        if self.mode=='PUBLIC_USER':
            if not self.structural or self.frozen is not None:
                yield None;return
            self._user_integrity();previous=self.phase;self.phase='OOF'
            try:yield 'AUTHENTIC_USER_TRAIN_FOLDS_NOT_INDEPENDENT_SCIENTIFIC_VALIDATION'
            finally:self.phase=previous
            return
        yield ('SOFTWARE_FOLD_FIXTURE_NOT_TRAINED_OOF'
               if type(self) is Session and self.mode=='FIXTURE' and self.structural else None)

    def _produce_oof_context(self, selected, program, evidence):
        self.check();start=time.monotonic()
        from .integrated import learning_program
        from .sampling import oof_folds
        from .preprocessing import TrainEncoder
        from morphtree.diagnostics import ErrorMemory, ErrorProfiler
        import math
        a=self.prepared
        if a is None:return None
        task=a['task']
        if task!='binary' and not callable(getattr(ErrorProfiler,'profile_task',None)):
            return None  # Older native runtimes keep their binary-only boundary.
        if program!=learning_program(selected) or selected['params']!=self.plan['params']:
            raise ValueError('OOF requires exact current fixed program and numeric parameters')
        raw=[a['rows'][i] for i in a['ti']];y=[a['y'][i] for i in a['ti']]
        ids=['ROW:'+str(i) for i in a['ti']]
        parent_encoder=TrainEncoder.from_dict(selected['recipe']['encoder'])
        encoded=parent_encoder.transform(_NamedRows(raw,a['names']))
        if digest({'X':encoded,'y':y,'ids':a['ti']})!=selected['learning_identity']['TRAIN']:
            raise ValueError('OOF parent differs from actual fixed TRAIN')
        try:held_folds=oof_folds(y,task,self.plan['seed'])
        except ValueError:
            self.event('OOF_ABSTAIN',reason='INSUFFICIENT_TRAIN_CLASS_OR_ROW_SUPPORT',task=task)
            return None
        self.admit_batch(task,selected['family'],selected['params'],folds=len(held_folds),kind='OOF')
        pred=[None]*len(y);proofs=[]
        # Write the plan before component intents. No SEARCH labels/predictions
        # enter folds, feature fitting or the profiler.
        key=digest({'parent':selected,'program':program})
        plan={'parent_sha256':digest(selected),'program_sha256':digest(program),
              'TRAIN_ids':ids,'held_folds':held_folds,'mode':self.mode,
              'evidence':evidence,'fold_local_preprocessing':True}
        path=artifact_path('OOF-'+key+'.json')
        if path.exists():
            if cache._read(path)!=plan:raise ValueError('Changed fixed-parent OOF plan')
        else:cache._write(path,plan)
        for held in held_folds:
            self.check();fit=[i for i in range(len(y)) if i not in set(held)]
            if not fit or not held or (task!='regression' and set(y[i] for i in fit)!=set(y)):
                return None  # Do not repair labels or leak held rows into fit.
            e=TrainEncoder().fit([raw[i] for i in fit]);e.state_['names']=a['names']
            fx=e.transform(_NamedRows([raw[i] for i in fit],a['names']))
            hx=e.transform(_NamedRows([raw[i] for i in held],a['names']))
            fy=[y[i] for i in fit];fold_ids=[ids[i] for i in fit]
            identity={'preprocessing':e.to_dict(),'seed':self.plan['seed'],'task':task,
                'family':selected['family'],'TRAIN_ids':fold_ids,
                'actual_TRAIN_X':fx,'actual_TRAIN_y':fy}
            # The owned worker validates this exact raw complement/encoder and
            # sample again before a native entry. It is not caller admission.
            self._fold_scope={'fit':fit,'held':held,'TRAIN_ids':ids,'encoder':e.to_dict(),
                'X':fx,'y':fy,'ids':fold_ids,'parent':copy.deepcopy(selected),
                'program':copy.deepcopy(program),'family':selected['family']}
            self._oof_capture=[]
            try:
                model=self.executor(None).fit_family(task,selected['family'],fx,fy,
                    seed=self.plan['seed'],params=HyperParams(**selected['params']),
                    genome=AlgorithmGenome(**program['genome']),check=self.check,identity=identity)
                receipts=copy.deepcopy(self._oof_capture)
            finally:self._oof_capture=None;self._fold_scope=None
            values=model.probabilities_or_values(hx)
            for i,v in zip(held,values):pred[i]=v
            proofs.append({'fit':fit,'held':held,'encoder':e.to_dict(),'receipts':receipts,
                           'held_X':hx,'prediction':values})
        from .support import prediction_contract
        prediction_contract(task,y,pred)
        px=[[float('nan') if v is None else v for v in row] for row in encoded]
        if task=='binary':
            memory=ErrorMemory(len(y));memory.update(y,pred)
            profile=ErrorProfiler().profile(px,y,memory,px,y,pred)
        else:profile=ErrorProfiler().profile_task(px,y,pred,task)
        profile['interpretation']='Current-program TRAIN-fold associations;not causal or unbiased evidence for the complete adaptive selection procedure'
        profile['acquisition_evidence']=evidence
        allowed=[]
        if program['genome']==AlgorithmGenome().to_dict():
            allowed=[digest(learning_program({'task':task,'family':selected['family'],
                'recipe':{'genome':AlgorithmGenome(growth_policy='depthwise').to_dict()}}))]
        context={'role':'TRAIN_OOF','parent_packet_sha256':digest(selected),
            'program_sha256':digest(program),'learning_identity':copy.deepcopy(selected['learning_identity']),
            'profile':profile,'allowed_programs':allowed,
            'evidence':evidence}
        proof={'context':copy.deepcopy(context),'parent':copy.deepcopy(selected),
            'plan':path.name,'plan_sha256':sha(path),'folds':proofs,'mode':self.mode,
            'acquisition_seconds_operational':time.monotonic()-start,
            'cost_scope':'INCLUSIVE_OPERATIONAL_NOT_PERFORMANCE','CPU':'UNKNOWN','peak_RSS':'UNKNOWN'}
        durable=artifact_path('OOF-PROOF-'+digest(context)+'.json');cache._write(durable,proof)
        self._diagnostic_proofs[digest(context)]={'context':copy.deepcopy(context),
            'parent':copy.deepcopy(selected),'artifact':durable.name,'sha256':sha(durable)}
        self.event('OOF_CONTEXT_PRODUCED',parent=digest(selected),context=digest(context),
                   evidence=evidence,components=sum(len(f['receipts']) for f in proofs),
                   physical_fits_scope=self.mode)
        return context

    def verify_evolution_context(self, context, selected):
        """Reopen committed components and reproduce every held prediction.

        No caller-provided boolean or metadata hash establishes authenticity.
        The proof is private to this exact software session; trained LIVE OOF
        remains blocked rather than inheriting this fixture qualification.
        """
        self.check()
        if self.mode=='FIXTURE':
            if context.get('evidence')!='SOFTWARE_FOLD_FIXTURE_NOT_TRAINED_OOF':return False
        elif self.mode=='PUBLIC_USER':
            if type(self) is not Session or context.get('evidence')!='AUTHENTIC_USER_TRAIN_FOLDS_NOT_INDEPENDENT_SCIENTIFIC_VALIDATION':return False
        elif self.mode=='LIVE':
            from .producer import Session as Owned, evolution_profile
            if type(self) is not Owned or not evolution_profile():return False
            self.permit()
            if context.get('evidence')!='AUTHENTIC_FIXED_PROGRAM_TRAIN_FOLDS_ADAPTIVE_UTILITY_UNPROVED':return False
        else:return False
        proof=self._diagnostic_proofs.get(digest(context))
        if proof is None or proof['context']!=context or proof['parent']!=selected:return False
        durable=artifact_path(proof['artifact'])
        if sha(durable)!=proof['sha256']:raise ValueError('Changed durable diagnostic proof')
        proof=cache._read(durable)
        if proof['mode']!=self.mode or proof['context']!=context or proof['parent']!=selected:return False
        if sha(artifact_path(proof['plan']))!=proof['plan_sha256']:raise ValueError('Changed OOF plan')
        from .support import FamilyModel
        from .preprocessing import TrainEncoder
        a=self.prepared;raw=[a['rows'][i] for i in a['ti']];y=[a['y'][i] for i in a['ti']]
        covered=[]
        for fold in proof['folds']:
            self.check();models=[]
            fit,held=fold['fit'],fold['held'];covered+=held
            if set(fit)&set(held) or sorted(fit+held)!=list(range(len(raw))):raise ValueError('Not a TRAIN-only complement')
            e=TrainEncoder().fit([raw[i] for i in fit]);e.state_['names']=a['names']
            if e.to_dict()!=fold['encoder'] or e.transform(_NamedRows([raw[i] for i in held],a['names']))!=fold['held_X']:
                raise ValueError('Fold-local preprocessing or held observations changed')
            for row in fold['receipts']:
                request=cache.FitRequest(cache.canonical(row['request']))
                v=request.value()
                expected=cache.learning_request(
                    {'role':'TRAIN','ids':['ROW:'+str(a['ti'][i]) for i in fit],
                     'X':e.transform(_NamedRows([raw[i] for i in fit],a['names'])),
                     'y':[y[i] for i in fit]},v['indices'],
                    **{k:v[k] for k in ('protocol','split_sha256','preprocessing','task','family',
                       'structure','params','config','objective','seed','backend','dependencies','sources')})
                if request!=expected or v['structure']!=selected['recipe']['genome'] or v['family']!=selected['family']:
                    raise ValueError('Fold component does not bind the actual TRAIN complement/current program')
                receipt=self.store._result(request,artifact_path(row['folder']))
                if digest(receipt)!=row['receipt_sha256']:raise ValueError('Changed OOF committed receipt')
                if request.value()['dependencies']['mode']!=self.mode:raise ValueError('Cross-mode OOF cache provenance')
                if self.mode in ('LIVE','PUBLIC_USER') and not any(r['event']=='NATIVE_FIT_COMPLETED' and r.get('phase')=='OOF' and r.get('key')==request.key for r in self.store.events()):
                    raise ValueError('No authenticated completed OOF native call')
                models.append(validate_model(artifact_path(receipt['artifact']['path']),request.value()))
            if not models:raise ValueError('Missing actual fold components')
            model=FamilyModel(selected['task'],models,[1/len(models)]*len(models))
            if model.probabilities_or_values(fold['held_X'])!=fold['prediction']:
                raise ValueError('OOF prediction differs from committed native component')
        if sorted(covered)!=list(range(len(raw))):raise ValueError('Held coverage differs')
        return True

    def _capture_oof_component(self, request, lease):
        if getattr(self,'_oof_capture',None) is not None:
            self._oof_capture.append({'request':request.value(),
                'folder':lease.folder.relative_to(self.root).as_posix(),'receipt_sha256':digest(lease.result)})

    def bind_input(self, estimator, X, y):
        self.check(); rows,names=_m26_rows(X); task,targets,classes=detect_target(y,estimator.task)
        if estimator.random_state!=self.plan['seed'] or self.mode!='PUBLIC_USER' and estimator.capacity is not None:
            raise ValueError('Exact seed/profile capacity required')
        if self.mode=='PUBLIC_USER':
            expected=asdict(HyperParams(n_estimators=2,max_leaves=8,max_depth=4,max_bins=15,min_samples_leaf=2,row_fraction=1.))
            if estimator.capacity:expected.update(estimator.capacity)
            if expected!=self.plan['params']:raise ValueError('Public requested capacity differs from immutable session')
        source=self.plan['source']
        if len(rows)!=len(source['rows']) or any(r!={'id':'ROW:'+str(i),'X':rows[i],'y':y[i]} for i,r in enumerate(source['rows'])):
            raise ValueError('Actual supplied data differ from admitted source')
        ti,si=_split(targets,task,self.plan['seed'])
        if source['partitions']!={'TRAIN':['ROW:'+str(i) for i in ti],
                                 'SEARCH':['ROW:'+str(i) for i in si],'GUARD_FINAL':[]}:
            raise ValueError('Actual ordered split differs')
        if source['OOF']: raise ValueError('No OOF producer admitted in COMP002')
        self.prepared={'rows':rows,'names':names,'y':targets,'classes':classes,'task':task,'ti':ti,'si':si}

    def parameters(self, unused):
        return HyperParams(**self.plan['params'])

    @property
    def source(self):return self.plan['source']

    @contextmanager
    def orchestration_operation(self, estimator):
        a=self.prepared; p=self.plan['params']; k=len(a['classes']) or 1
        families=['boost','bag'] if estimator.families=='auto' else [estimator.families]
        forms=[{'family':f,'members':1 if f=='boost' else p['n_estimators'],
                'stages':p['n_estimators'] if f=='boost' else 1,'classes':k,
                'genome_sha256':digest(AlgorithmGenome().to_dict())} for f in families]
        form=forms[0] if len(forms)==1 else {'candidates':forms}
        if hasattr(self,'admission_recipe'):
            form=self.admission_recipe(estimator,form)
        request=self.engine.request(operation='inspect',task=a['task'],recipe=form,
            source=self.source,role='TRAIN',protocol=self.engine.q['protocol'],
            authorization=self.engine.q['authorization']['id'],components=[])
        # One exact software operation contains the canonical per-component
        # intents. This inspect capability never itself authorizes training.
        with self.engine.operation(request): yield

    def optional_optuna(self):
        if self.mode!='PUBLIC_USER':return False
        import importlib.util
        return importlib.util.find_spec('optuna') is not None
    def numeric_admitted(self, executor):
        return self.mode=='PUBLIC_USER' and self._user_limits['trials']>0
    def load_optuna(self):
        if self.mode=='PUBLIC_USER':
            if not self.numeric_admitted(None):raise PermissionError('No numerical user budget')
            import importlib
            from .producer import NumericBinding
            module=importlib.import_module('optuna')
            from importlib.metadata import distribution
            dist=distribution('optuna')
            if module.__version__!='5.0.0' or dist.version!='5.0.0' or not Path(module.__file__).resolve().is_relative_to(Path(dist.locate_file('optuna')).resolve()):
                raise ValueError('Exact supported Optuna5.0.0 distribution required;no shadowed driver')
            return NumericBinding(self,module)
        raise AdmissionError({'status':'BLOCKED','reason':'OPTIONAL_DEPENDENCY_AND_LIVE_TRIAL_PROFILE_REQUIRED'})

    def reserve_trial(self, *, real_driver=True):
        self._learning_open()
        if self.mode!='PUBLIC_USER' or self.frozen is None or self.phase!='NUMERIC':
            raise PermissionError('User trial requires an already frozen program')
        if type(real_driver) is not bool:raise ValueError('Explicit driver evidence required')
        self.check()
        if self._user_trials>=self._user_limits['trials']:raise RuntimeError('Numerical user budget exhausted')
        if self.public_scope is not None:self.public_scope.reserve(self,operation='trial',trees=0,
            key=digest({'frozen':self.frozen,'ordinal':self._user_trials,'source':digest(self.source)}))
        r=self._user_learning_request('trial',self.frozen['task'],self.frozen['family'],
            HyperParams(**self.frozen['params']),len(self.frozen['recipe']['classes']),self.frozen['recipe']['genome'])
        with self.engine.user_learning(r):pass  # Reservation precedes Optuna.ask, not a completed numerical trial.
        self._user_trials+=1
        self.event('NUMERIC_TRIAL_INTENT',real=real_driver,frozen_sha256=digest(self.frozen))

    def _user_learning_request(self,operation,task,family,params,classes,genome):
        recipe={'family':family,'members':params.n_estimators if family=='bag' and operation=='trial' else 1,'stages':params.n_estimators if family=='boost' else 1,
            'classes':classes if task!='regression' else 1,'genome_sha256':digest(genome)}
        r=self.engine.request(operation=operation,task=task,recipe=recipe,source=self.source,role='TRAIN',
            protocol=self.engine.q['protocol'],authorization=self.engine.q['authorization']['id'],components=[],
            requested_cost={'fits':int(operation=='fit'),'trials':int(operation=='trial')})
        r['component_key']=digest({'operation':operation,'ordinal':self._user_started if operation=='fit' else self._user_trials,'recipe':recipe})
        r['id']=digest({k:v for k,v in r.items() if k!='id'})
        return r

    def packet_lineage(self,genome):
        if self.mode=='PUBLIC_USER' and self.frozen is not None:
            if genome.to_dict()!=self.frozen['recipe']['genome']:raise ValueError('Changed frozen numerical genome')
            return copy.deepcopy(self.frozen['recipe'].get('lineage',{}))
        return {}

    def _user_integrity(self):
        if self.mode!='PUBLIC_USER':raise PermissionError('Ordinary-user boundary only')
        self.check()
        configure(self.root);cache.configure(self.root,self.engine.package)
        if digest(self.plan)!=self.pin or sha(self.root/'SESSION.json')!=self.policy_sha or digest(self.engine.q)!=self.engine._q_pin:
            raise ValueError('Mutated ordinary-user plan/qualification')
        if installed_identity(self.engine.package,self.engine.q)!=self.engine.identity:
            raise ValueError('Changed installed ordinary-user runtime')
        if self._user_limits!=self.plan['user_limits']:raise ValueError('Changed ordinary-user resource limits')

    def close_user_run(self):
        if self.mode!='PUBLIC_USER' or self._closed:return
        self.event('USER_RUN_CLOSED',counts=self.counters(),authority_residual=0)
        self._closed=True
        self.engine.stop_permanently('Ordinary user call finished;never a scientific training permit')

    def executor(self, estimator):
        session=self
        class Executor(NativeExecutor):
            evidence_level={'PUBLIC_USER':'USER_API_NATIVE_EXECUTION_NOT_SCIENTIFIC_CONFIRMATION','FIXTURE':'COMP003_SOFTWARE_FIXTURE_NOT_TRAINING',
                'PREFIT':'COMP003_PREFIT_NOT_TRAINING','LIVE':'COMP003_EXACT_OWNER_ADMITTED_NATIVE_TRAINING'}[session.mode]
            def execute_native(self, model, X, y, *, identity, cancel):
                return session.execute_native(model,X,y,identity=identity,cancel=cancel)
        return Executor()

    def execute_native(self, model, X, y, *, identity, cancel):
        start=time.monotonic();before=getattr(self,'_user_completed',0)
        try:return self._execute_native(model,X,y,identity=identity,cancel=cancel)
        finally:
            if self.mode=='PUBLIC_USER' and self._user_completed>before:
                elapsed=time.monotonic()-start
                self._component_costs.append(elapsed)
                self.event('USER_COMPONENT_OPERATION_COST',seconds=elapsed,
                    scope='INCLUSIVE_COMPONENT_OPERATION_ONLY_NOT_TOTAL_FIT_OR_BENCHMARK')

    def _execute_native(self, model, X, y, *, identity, cancel):
        self._learning_open()
        if self.mode=='PUBLIC_USER':
            self._user_integrity()
            if self.phase=='NUMERIC':
                if self.frozen is None or identity['family']!=self.frozen['family'] or model.base_genome.to_dict()!=self.frozen['recipe']['genome']:
                    raise ValueError('Numerical tuning cannot change the frozen program')
                base=HyperParams(**self.frozen['params'])
                if identity['family']=='bag':base=replace(base,n_estimators=1,learning_rate=1.,row_fraction=1.)
                allowed={'l2','min_samples_leaf'}|({'learning_rate'} if identity['family']=='boost' else set())
                if any(asdict(model.base_params)[k]!=v for k,v in asdict(base).items() if k not in allowed):
                    raise ValueError('Numerical tuning changed a fixed parameter')
        self.check(); a=self.prepared; task=a['task']; family=identity['family']
        if type(model) is not NATIVE[task] or model.backend!='pure' or model.model_ is not None:
            raise ValueError('Fresh exact native estimator required')
        p=asdict(model.base_params); c=asdict(model.config)
        genome=model.base_genome.to_dict()
        train={'role':'TRAIN','ids':identity['TRAIN_ids'],
               'X':identity['actual_TRAIN_X'],'y':identity['actual_TRAIN_y']}
        indices=identity['ordered_sample']
        if X!=[train['X'][i] for i in indices] or y!=[train['y'][i] for i in indices] or p!=identity['params']:
            raise ValueError('Actual sampled data/parameters differ')
        prep={**identity['preprocessing'],'fit_role':'TRAIN','encoded_classes':list(range(len(a['classes'])))}
        component_protocol={k:self.plan['protocol'][k] for k in ('path','sha256')}
        context=dict(protocol=component_protocol,split_sha256=digest(self.source['partitions']),
            preprocessing=prep,task=task,family=family,structure=genome,params=p,config=c,
            objective={'binary':'binary_logloss','multiclass':'ovr','regression':'squared_error'}[task],
            seed=c['seed'],backend='pure',dependencies={'runtime':self.engine.identity,'mode':self.mode},
            sources=cache.native_sources())
        request=cache.learning_request(train,indices,**context)
        form={'family':family,'members':1,'stages':p['n_estimators'],'classes':len(a['classes']) or 1,'genome_sha256':digest(genome)}
        from .admission import CURRENT
        current=CURRENT.get()
        if current is None or current[0] is not self.engine or current[1]['operation']!='inspect':
            raise ValueError('Missing exact parent software admission')
        if True:
            self.logical+=1
            if self.logical>self.plan['max_requests']: raise RuntimeError('Logical opportunity budget exhausted')
            with self.store.claim(request,logical_id='component:'+str(self.logical)) as lease:
                if lease.hit:
                    self._capture_oof_component(request,lease)
                    return validate_model(artifact_path(lease.result['artifact']['path']),request.value())
                if self.mode in ('PREFIT','LIVE','PUBLIC_USER'):
                    self.component=(model,copy.deepcopy(X),copy.deepcopy(y),request)
                    try:
                        native=model.fit(X,y,cancel=lambda:(cancel() or False))
                        path=artifact_path(lease.folder/'model.json');native.save_model(path)
                        validated=validate_model(path,request.value())
                        if self.mode=='PUBLIC_USER':
                            self._user_integrity()
                            if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())>self._user_limits['artifact_bytes']:
                                raise RuntimeError('Ordinary-user artifact budget exhausted')
                        lease.commit(path,expected_sha256=sha(path),proof=[component_protocol])
                        self._capture_oof_component(request,lease)
                        return validated
                    finally:self.component=None
                self.fixture_calls+=1
                payload=self.fixture(copy.deepcopy(request.value()),copy.deepcopy(X),copy.deepcopy(y))
                path=artifact_path(lease.folder/'model.json')
                cache._write(path,payload)
                native=validate_model(path,request.value())
                lease.commit(path,expected_sha256=sha(path),proof=[component_protocol])
                self._capture_oof_component(request,lease)
                self.event('FIXTURE_COMPLETED',key=request.key,physical_fits=0)
                return native

    def native_boundary(self, original, model, X, y, kwargs):
        self._learning_open()
        self.check()
        if self.mode=='PUBLIC_USER':
            self._user_integrity()
            if self.component is None:raise PermissionError('No active user component lease')
            want,xx,yy,request=self.component;v=request.value()
            if model is not want or X!=xx or y!=yy or set(kwargs)!={'cancel'} or not callable(kwargs['cancel']):
                raise ValueError('Actual user native invocation differs')
            if asdict(model.base_params)!=v['params'] or asdict(model.config)!=v['config'] or model.base_genome.to_dict()!=v['structure']:
                raise ValueError('Actual user native recipe changed')
            scalar=v['params']['n_estimators']*(len(v['preprocessing']['encoded_classes']) if v['task']=='multiclass' else 1)
            if self._user_started+1>self._user_limits['calls'] or self._user_trees+scalar>self._user_limits['scalar_trees']:
                raise RuntimeError('Ordinary-user cumulative native budget exhausted')
            r=self._user_learning_request('fit',v['task'],v['family'],model.base_params,
                len(v['preprocessing']['encoded_classes']),v['structure'])
            # Repeated equal parameters still need distinct physical component
            # identities. Cache hits never reach this reservation boundary.
            r['component_key']=request.key
            r['id']=digest({k:x for k,x in r.items() if k!='id'})
            if self.public_scope is not None:self.public_scope.reserve(self,operation='fit',trees=scalar,key=request.key)
            with self.engine.user_learning(r):
                self._user_started+=1;self._user_trees+=scalar;self.physical+=1
                self.event('NATIVE_FIT_STARTED',key=request.key,scalar_trees=scalar,physical_fits=1)
                old=getattr(self,'_component_deadline',None)
                self._component_deadline=min(time.monotonic()+60,self.started+self.plan['seconds'])
                try:
                    fitted=original(model,X,y,**kwargs)
                    self._user_integrity();self._user_completed+=1
                    self.event('NATIVE_FIT_COMPLETED',key=request.key,scalar_trees=scalar,physical_fits=1)
                    return fitted
                except BaseException as e:
                    self.event('NATIVE_FIT_INCOMPLETE',key=request.key,error=type(e).__name__)
                    raise
                finally:self._component_deadline=old
        if self.mode!='PREFIT' or self.component is None:
            raise AdmissionError({'status':'BLOCKED','reason':'NO_CURRENT_LIVE_NATIVE_AUTHORITY'})
        want,xx,yy,request=self.component
        v=request.value()
        if model is not want or X!=xx or y!=yy or set(kwargs)!={'cancel'} or not callable(kwargs['cancel']):
            raise ValueError('Actual pre-fit invocation differs')
        if asdict(model.base_params)!=v['params'] or asdict(model.config)!=v['config']:
            raise ValueError('Changed actual pre-fit parameters/config')
        self.prefit+=1;self.event('NATIVE_PREFIT_BOUNDARY',key=request.key,physical_fits=0)
        raise PreFitBoundary('STOP_BEFORE_ORIGINAL_NATIVE_FIT')


def native_fit_entry(original, cls):
    """M031 cooperative pre-fit wrapper, with exact current v2 session type."""
    def guarded(model,X,y,**kwargs):
        session=require_session()
        if type(model) is not cls: raise ValueError('Changed native class')
        return session.native_boundary(original,model,X,y,kwargs)
    return guarded
