"""COMP003 profile adapter to the existing installed worker and Session.

Not an authorization framework: owner claim/permit, pipe handoff, OS leases,
component cache, native execution and finalizers are the existing components.
The exact finite profile below is the only newly qualified campaign scope.
"""
import copy
import importlib
import importlib.util
import json
import sys
import time
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from morphtree.config import AlgorithmGenome, HyperParams
from .admission import AdmissionError, digest, sha, read, safe_path, installed_identity
from .runtime_io import configure, artifact_path, publish, reference, WORKDIR
from .execution_v2 import Session as ExistingSession, PreFitBoundary
from .target import _NamedRows, _split, detect_target
from .preprocessing import TrainEncoder, _m26_rows
from .sampling import bootstrap
from .processes import exclusive_lock
from . import component_cache_v2 as cache

PROFILE={'id':'COMP009_NATIVE_EVOLUTION_V1','calls':13,'scalar_trees':14,'trials':0,
    'global_seconds':600,'worker_seconds':540,'component_seconds':60,
    'workers':1,'artifact_bytes':32*1024**2,'OOF':8,'GUARD':0,'refits':0,'retries':0}
PHASES=('FAMILY','OOF','STRUCTURE','NUMERIC','FINALIZATION')
CAPS={'FAMILY':{'calls':3,'trees':4,'trials':0},
      'OOF':{'calls':8,'trees':8,'trials':0},
      'STRUCTURE':{'calls':2,'trees':2,'trials':0},
      'NUMERIC':{'calls':0,'trees':0,'trials':0}}
PIPE_CURRENT=ContextVar('comp009-owned-evolution-pipe',default=None)
WORKFLOWS=[('binary','auto','auto','none')]


def evolution_profile():
    return PROFILE.get('id')=='COMP009_NATIVE_EVOLUTION_V1'


def oof_profiles(case,params,seed):
    """Four exact fixed-program declarations, not four independent sources.

    Stored by the existing AdmissionEngine as explicit TRAIN-fold sources.
    Nothing is inferred from a role name or mutable caller-supplied indices.
    """
    from .sampling import oof_folds
    from .integrated import learning_program
    raw=case['raw'];_,y,_=detect_target(raw['y'],'binary')
    ty=[y[i] for i in case['TRAIN']];ids=case['source']['partitions']['TRAIN']
    held=oof_folds(ty,'binary',seed);result={}
    for family in ('boost','bag'):
        for genome in (AlgorithmGenome(),AlgorithmGenome(growth_policy='depthwise')):
            program=learning_program({'task':'binary','family':family,'recipe':{'genome':genome.to_dict()}})
            frozen=digest({'program':program,'params':params,'seed':seed})
            source=copy.deepcopy(case['source'])
            source['OOF']=[{'fit':[ids[i] for i in range(len(ids)) if i not in set(h)],
                            'held':[ids[i] for i in h],'recipe_sha256':frozen} for h in held]
            result[digest(program)]={'program':program,'params':copy.deepcopy(params),'seed':seed,'source':source}
    return result


def bound_reference(root, ref):
    if type(ref) is not dict or set(ref)!={'path','bytes','sha256'}:raise ValueError('Exact supplied reference')
    p=safe_path(root,ref['path'])
    if p.stat().st_size!=ref['bytes']:raise ValueError('Changed bound size')
    return read(p,expected_sha256=ref['sha256'])


def prepare(engine,plan):
    evo=evolution_profile()
    keys={'schema','id','protocol','profile','params','seed','cases','max_requests'}|({'oof_profiles','max_mutations'} if evo else set())
    if set(plan)!=keys or plan['schema']!=('comp009-evolution-v1' if evo else 'comp003-multiphase-v2'):
        raise ValueError('Exact multiphase profile required')
    if plan['id']!=('COMP009-LIVE-001' if evo else 'COMP009-LIVE-001') or plan['profile']!=PROFILE or plan['protocol']!=engine.q['protocol'] or plan['max_requests']!=32:
        raise ValueError('Unqualified campaign/profile/protocol')
    wanted=asdict(HyperParams(n_estimators=2,max_leaves=8,max_depth=4,max_bins=15,min_samples_leaf=2,row_fraction=1.))
    if plan['params']!=wanted or plan['seed']!=42 or set(plan['cases'])!=({'binary'} if evo else {'binary','multiclass','regression'}):
        raise ValueError('Frozen INT002 technical recipes/seed only')
    result={}
    for task,c in plan['cases'].items():
        if set(c)!={'dataset','source'}:raise ValueError('Explicit data/source')
        raw=bound_reference(engine.q['input_root'],c['dataset'])
        rows,names=_m26_rows(_NamedRows(raw['X'],raw['names']));actual,y,classes=detect_target(raw['y'],task)
        if actual!=task or len(rows)!=48 or len(rows[0])!=3:raise ValueError('Exact small exposed technical source')
        source=c['source']
        if source['origin']['original_sha256']!=c['dataset']['sha256'] or source['exposure']!='DEV_EXPOSED' or source['sealed'] or source['OOF']:
            raise ValueError('Not the admitted exposed source')
        if source['rows']!=[{'id':'ROW:'+str(i),'X':row,'y':raw['y'][i]} for i,row in enumerate(rows)]:raise ValueError('Changed source content')
        ti,si=_split(y,task,plan['seed'])
        if source['partitions']!={'TRAIN':['ROW:'+str(i) for i in ti],'SEARCH':['ROW:'+str(i) for i in si],'GUARD_FINAL':[]}:
            raise ValueError('Changed actual frozen split')
        if engine.q['sources'].get(digest(source))!=source:raise ValueError('Unknown source')
        encoder=TrainEncoder().fit([rows[i] for i in ti]);encoder.state_['names']=names
        if encoder.width_!=7:raise ValueError('Expected exact retained encoder width')
        result[task]={'raw':raw,'source':source,'classes':classes,'TRAIN':ti,'SEARCH':si,'encoder':encoder.to_dict()}
    if evo:
        from .admission import validate_roles
        if plan['max_mutations']!=2 or plan['oof_profiles']!=oof_profiles(result['binary'],wanted,42):
            raise ValueError('Exact fixed-parent program/fold declarations required')
        for profile in plan['oof_profiles'].values():
            s=profile['source'];validate_roles(s,engine.q['sources'].get(digest(s)))
    key=digest({'runtime':engine.identity,'plan':plan,'actual_cases':result})
    return {'plan':copy.deepcopy(plan),'key':key,'task':'multiphase','cases':result,
        'request':{'learning_budget':{k:PROFILE[k] for k in ('calls','scalar_trees','trials')}}}


def optional_module(engine, *, load):
    lock=engine.q.get('optional_dependency')
    if lock is None:return None
    if set(lock)!={'root','files','version'} or lock['version']!='5.0.0' or not lock['files']:raise ValueError('Exact optional source lock')
    root=Path(lock['root']).resolve()
    if root!=engine.package.parent:raise ValueError('Optional dependency must be installed beside this wheel, no checkout path')
    for name,h in lock['files'].items():
        if Path(name).parts[0]=='morphtree' or sha(safe_path(root,name))!=h:raise ValueError('Changed/shadowing optional dependency')
    spec=importlib.util.find_spec('optuna')
    if spec is None:return None
    if Path(spec.origin).resolve()!=root/'optuna/__init__.py':raise ValueError('Unqualified optional origin')
    if not load:return lock
    module=importlib.import_module('optuna')
    if module.__version__!=lock['version']:raise ValueError('Wrong optional version')
    return module


class NumericBinding:
    """Account before ask; reuse real Optuna only in the owner-admitted worker."""
    def __init__(self, session, module):
        self.session=session;self.module=module;self.samplers=module.samplers
        self.trial=module.trial
        self.actual_driver=getattr(module,'__name__',None)=='optuna'
    def create_study(self, **kwargs):
        s=self.session;s.check()
        if s.frozen is None:raise ValueError('No study before structural freeze')
        study=self.module.create_study(**kwargs)
        class Study:
            def ask(self):
                if s.mode=='PUBLIC_USER':s.reserve_trial(real_driver=self_outer.actual_driver)
                else:s.reserve_trial()
                result=study.ask()
                if s.mode=='PUBLIC_USER':s.event('NUMERIC_TRIAL_ASKED',number=result.number,real=self_outer.actual_driver)
                return result
            def tell(self,trial,*args,**kw):
                # Failed/interrupted trials remain charged even if tell fails.
                s.check();value=study.tell(trial,*args,**kw)
                s.event('NUMERIC_TRIAL_COMPLETED',number=trial.number,
                    real=(self_outer.actual_driver if s.mode=='PUBLIC_USER' else s.mode=='LIVE'),state='FAIL' if kw else 'COMPLETE')
                return value
        self_outer=self
        return Study()


class Session(ExistingSession):
    """Specific M031 pipe-owned multiphase adapter, one shared canonical store."""
    def admission_recipe(self, estimator, form):
        # This is the actual additional native form that this workflow can
        # acquire, not an arbitrary tag used to mint another capability.
        if evolution_profile():
            return {'schema':'comp009-evolution-capacity-v1','initial':form,
                    'OOF':{'calls':8,'scalar_trees':8,'folds':2,'max_contexts':2},
                    'child_genome':AlgorithmGenome(growth_policy='depthwise').to_dict()}
        if estimator._research_growth and self.structural:
            if set(form)!={'candidates'} or self.prepared['task']!='binary':
                raise ValueError('Explicit binary two-family research scope required')
            return {**form,'research_genome':AlgorithmGenome(growth_policy='depthwise').to_dict()}
        return form

    def __init__(self,engine,plan,policy,root,*,owner_permit=None,owner_text_sha=None):
        configure(root);self.root=artifact_path(root);self.engine=engine
        if self.root!=engine.workdir or self.root!=Path(engine.q['input_root']).resolve():raise ValueError('Exact supplied scope')
        self.policy=copy.deepcopy(policy);self.policy_pin=digest(policy)
        if PIPE_CURRENT.get()!=self.policy_pin:raise PermissionError('No actual pre-operation pipe release')
        self.policy_file_pin=sha(self.root/'RUN-POLICY.json')
        self.plan=copy.deepcopy(plan);self.pin=digest(plan);self.campaign=prepare(engine,plan)
        self.mode=policy['mode'];self.owner=owner_permit;self.owner_text_sha=owner_text_sha
        if self.mode not in ('LIVE','PREFIT','FIXTURE'):raise ValueError('Known finite mode')
        self.fixture=self.fixture_payload if self.mode=='FIXTURE' else None
        self.started=time.monotonic();self.prepared=None;self.physical=0;self.prefit=0;self.fixture_calls=0
        self.logical=0;self.component=None;self.structural=False;self.cursor=0;self.phase=None;self.frozen=None;self.native_started=None
        self._oof_capture=None;self._fold_scope=None;self._diagnostic_proofs={};self._oof_parent=None
        self._pending_program=None;self._oof_context_intents=0
        if (self.root/'SESSION.json').exists():raise RuntimeError('Consumed session; no new deadline/attempt')
        cache.configure(self.root,engine.package)
        cache._write(self.root/'SESSION.json',{'plan':self.plan,'policy':self.policy,'runtime':engine.identity})
        self.policy_sha=sha(self.root/'SESSION.json')
        self.store=cache.MultitaskStore(self.root/'store',max_compute_jobs=PROFILE['calls'],seconds=PROFILE['worker_seconds'])
        self.check()
        if self.mode=='LIVE':self.permit()

    def check(self):
        configure(self.root);cache.configure(self.root,self.engine.package)
        if PIPE_CURRENT.get()!=self.policy_pin or digest(self.policy)!=self.policy_pin or digest(self.plan)!=self.pin:
            raise ValueError('Mutated plan/policy/pipe')
        if sha(self.root/'RUN-POLICY.json')!=self.policy_file_pin or sha(self.root/'SESSION.json')!=self.policy_sha:raise ValueError('Changed durable policy')
        if digest(self.engine.q)!=self.engine._q_pin or installed_identity(self.engine.package,self.engine.q)!=self.engine.identity:raise ValueError('Changed qualification/runtime')
        if self.engine.stop.exists() or (self.root/'STOP').exists() or (self.store.root/'STOP').exists():raise RuntimeError('STOP permanent')
        self.store.events()  # Original chained journal rejects partial/corrupt tails.
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())>PROFILE['artifact_bytes']:
            raise RuntimeError('Global artifact cap exceeded')
        if time.time()>=self.policy['deadline_utc'] or time.monotonic_ns()>=self.policy['deadline_ns']:raise TimeoutError('Global immutable deadline')
        if self.native_started is not None and time.monotonic()-self.native_started>=PROFILE['component_seconds']:raise TimeoutError('Cooperative component deadline')

    @property
    def source(self):return self.prepared['source']

    def bind_input(self, estimator,X,y):
        self.check()
        if self.cursor>=len(WORKFLOWS):raise RuntimeError('All planned workflows consumed')
        task,family,morph,tuning=WORKFLOWS[self.cursor]
        c=self.campaign['cases'][task];raw=c['raw'];rows,names=_m26_rows(X)
        if rows!=raw['X'] or names!=raw['names'] or y!=raw['y'] or estimator.task not in ('auto',task):raise ValueError('Wrong ordered actual input/task')
        if (estimator.families,estimator.morphing,estimator.tuning)!=(family,morph,tuning) or estimator.random_state!=42 or estimator.capacity is not None:
            raise ValueError('Not the next frozen workflow')
        if tuning=='optuna' and estimator.n_trials!=2:raise ValueError('Two startup trials only')
        if evolution_profile() and (not estimator._evolving_program or estimator._research_growth or estimator._max_mutations!=2):
            raise ValueError('Exact automatic evolutionary API and finite two-decision scope required')
        _,targets,classes=detect_target(y,task)
        self.prepared={'task':task,'source':c['source'],'rows':rows,'y':targets,'classes':classes,'names':names,
                       'ti':c['TRAIN'],'si':c['SEARCH']}
        self.workflow_physical_start=self.store.counts()['NATIVE_FIT_STARTED']
        self.phase='FAMILY' if evolution_profile() else 'BASELINE' if self.cursor<3 else 'FAMILY'
        self.structural=morph=='auto';self.frozen=None;self.lineage={}
        self.cursor+=1

    def event(self,kind,**values):
        if kind=='ORCHESTRATION':
            e=values['value']
            if evolution_profile() and e['operation']=='evolution_intent':
                from .integrated import program_dispatch
                program=e['child_program'];family,genome=program_dispatch(program)
                if self.frozen is not None or e['gene']!='growth_policy' or genome!=AlgorithmGenome(growth_policy='depthwise'):
                    raise ValueError('Only the declared operative growth transformation before freeze')
                self.phase='STRUCTURE';self._pending_program=copy.deepcopy(program)
                self.lineage={'gene':e['gene'],'parent_packet_sha256':e['parent_packet_sha256'],
                    'from':AlgorithmGenome().to_dict(),'to':genome.to_dict(),
                    'reason':'OOF_HEURISTIC_HYPOTHESIS_NOT_CAUSAL_UTILITY'}
            if e['operation']=='structural_proposal':
                if self.frozen is not None:raise ValueError('No mutation after freeze')
                self.phase='STRUCTURE'
                self.lineage={'gene':e['gene'], 'parent_packet_sha256':digest(e['parent_packet']),
                    'from':copy.deepcopy(e['parent_packet']['recipe']['genome']), 'to':copy.deepcopy(e['genome']),
                    'reason':'STATIC_TECHNICAL_NOMINATION_NOT_DIAGNOSTIC_UTILITY'}
            if e['operation']=='structure_frozen':
                if 'packet' in e:
                    if self.frozen is not None and self.frozen!=e['packet']:raise ValueError('Changed frozen structure')
                    self.frozen=copy.deepcopy(e['packet'])
                self.phase='NUMERIC' if self.cursor==6 else 'FINALIZATION'
        self.store.event(kind,phase=self.phase,**values)
        if kind=='ORCHESTRATION' and values['value']['operation']=='structural_rollback':
            # A verified worse candidate is a normal ROLLBACK decision. This
            # event means an exception: keep the anchor, stop the campaign,
            # and never spend the subsequent stage's unused budget.
            raise RuntimeError('STRUCTURAL_PHASE_FAILED_TERMINAL_NO_NUMERIC_CONTINUATION')
        if kind=='ORCHESTRATION' and evolution_profile() and values['value']['operation']=='evolution_failure':
            raise RuntimeError('EVOLUTION_COMPONENT_FAILED_TERMINAL_ANCHOR_PRESERVED')

    @contextmanager
    def oof_operation(self,selected,program):
        self.check()
        if not evolution_profile() or not self.structural or self.frozen is not None:
            yield None;return
        if self.mode not in ('LIVE','FIXTURE','PREFIT'):raise PermissionError('No OOF mode')
        if self.mode=='LIVE':self.permit()
        profile=self.plan['oof_profiles'].get(digest(program))
        if profile is None or profile['program']!=program or selected['params']!=profile['params']:
            raise ValueError('Current program has no exact declared fold profile')
        if self._oof_context_intents>=2:raise RuntimeError('Two-context OOF opportunity budget exhausted')
        form={'family':selected['family'],'members':1 if selected['family']=='boost' else profile['params']['n_estimators'],
              'stages':profile['params']['n_estimators'] if selected['family']=='boost' else 1,
              'classes':2,'genome_sha256':digest(program['genome'])}
        r=self.engine.request(operation='inspect',task='binary',recipe=form,source=profile['source'],role='OOF',
            protocol=self.engine.q['protocol'],authorization=self.engine.q['authorization']['id'],components=[])
        previous=self.phase;self.phase='OOF';self._oof_parent=copy.deepcopy(selected)
        try:
            with self.engine.fold_operation(r):
                self._oof_context_intents+=1
                self.event('OOF_CONTEXT_INTENT',parent=digest(selected),program=digest(program),
                           fold_source=digest(profile['source']),role='TRAIN_FOLDS')
                yield ('AUTHENTIC_FIXED_PROGRAM_TRAIN_FOLDS_ADAPTIVE_UTILITY_UNPROVED' if self.mode=='LIVE'
                       else 'SOFTWARE_FOLD_FIXTURE_NOT_TRAINED_OOF' if self.mode=='FIXTURE' else 'PREFIT_NO_TRAINED_OOF')
        finally:self.phase=previous;self._oof_parent=None;self._fold_scope=None

    def _validate_evolution_input(self,model,identity):
        """Reconstruct raw complements; reject a superficial role/ID rewrite."""
        from .sampling import oof_folds
        c=self.campaign['cases']['binary'];raw=c['raw'];_,y,_=detect_target(raw['y'],'binary')
        ids=['ROW:'+str(i) for i in c['TRAIN']];rows=[raw['X'][i] for i in c['TRAIN']];ty=[y[i] for i in c['TRAIN']]
        if self.phase=='OOF':
            scope=self._fold_scope
            if scope is None or self._oof_parent is None:raise ValueError('Missing owned current fold')
            if scope['held'] not in oof_folds(ty,'binary',42) or scope['fit']!=[i for i in range(len(ty)) if i not in set(scope['held'])]:
                raise ValueError('Changed raw fold complement')
            fit=scope['fit'];rows=[rows[i] for i in fit];ty=[ty[i] for i in fit];ids=[ids[i] for i in fit]
            wanted=self._oof_parent['recipe']['genome'];family=self._oof_parent['family']
        elif self.phase=='STRUCTURE':
            if self._pending_program is None:raise ValueError('Missing typed child program')
            wanted=self._pending_program['genome'];family=self._pending_program['family']
        elif self.phase=='FAMILY':wanted=AlgorithmGenome().to_dict();family=identity['family']
        else:raise ValueError('No native learning outside family/OOF/structure phases')
        encoder=TrainEncoder().fit(rows);encoder.state_['names']=raw['names']
        encoded=encoder.transform(_NamedRows(rows,raw['names']))
        if identity['family']!=family or identity['TRAIN_ids']!=ids or identity['actual_TRAIN_y']!=ty or identity['actual_TRAIN_X']!=encoded or identity['preprocessing']!=encoder.to_dict():
            raise ValueError('Actual native TRAIN differs from owned raw complement/preprocessing')
        if model.base_genome.to_dict()!=wanted:raise ValueError('Actual native learning program mismatch')
        params=HyperParams(**self.plan['params'])
        if family=='bag':params=replace(params,n_estimators=1,learning_rate=1.,row_fraction=1.)
        if asdict(model.base_params)!=asdict(params):raise ValueError('Current fixed-parent numerical recipe mismatch')

    def packet_lineage(self, genome):
        if self.phase=='NUMERIC' and self.frozen is not None:
            return copy.deepcopy(self.frozen['recipe']['lineage'])
        return copy.deepcopy(self.lineage) if genome!=AlgorithmGenome() else {}

    def _used(self):
        rows=self.store.events()
        used={p:{'calls':0,'trees':0,'trials':0} for p in CAPS}
        for r in rows:
            if r['event'] in ('NATIVE_FIT_STARTED','FIXTURE_FIT_INTENT'):
                used[r['phase']]['calls']+=1;used[r['phase']]['trees']+=r['scalar_trees']
            if r['event']=='NUMERIC_TRIAL_INTENT':used['NUMERIC']['trials']+=1
        return used

    def reserve(self,*,calls=0,trees=0,trials=0):
        self.check();used=self._used()
        if self.phase not in CAPS:raise ValueError('No learning in this phase')
        add={'calls':calls,'trees':trees,'trials':trials}
        if any(used[self.phase][k]+v>CAPS[self.phase][k] for k,v in add.items()):raise RuntimeError('Subbudget exhausted')
        total={k:sum(r[k] for r in used.values()) for k in add}
        if total['calls']+calls>PROFILE['calls'] or total['trees']+trees>PROFILE['scalar_trees'] or total['trials']+trials>PROFILE['trials']:raise RuntimeError('Global budget exhausted')

    def reserve_trial(self):
        if self.frozen is None or self.phase!='NUMERIC':raise ValueError('No trial before freeze')
        if not self.engine.q.get('numeric_authorized') or self.engine.q.get('trial_cap',0)<=0:
            raise PermissionError('Zero/unapproved numeric budget')
        if self._used()['NUMERIC']['trials']>=self.engine.q['trial_cap']:raise RuntimeError('Numeric budget exhausted')
        if self.mode=='LIVE':self.permit()
        elif self.mode!='FIXTURE':raise PermissionError('No numeric work in PREFIT')
        self.reserve(trials=1)
        self.event('NUMERIC_TRIAL_INTENT',real=self.mode=='LIVE',frozen_sha256=digest(self.frozen))

    def optional_optuna(self):
        return optional_module(self.engine,load=False) is not None
    def numeric_admitted(self, executor):
        if not self.engine.q.get('numeric_authorized',False) or self.engine.q.get('trial_cap',0)<=0:return False
        if self.mode=='LIVE':self.permit()
        return self.mode in ('LIVE','FIXTURE')
    def load_optuna(self):
        if not self.numeric_admitted(None):raise PermissionError('No exact numeric admission')
        if self.mode=='FIXTURE':
            from .numeric_fixture import module
            return NumericBinding(self,module(self.engine.q.get('numeric_fault')))
        return NumericBinding(self,optional_module(self.engine,load=True))

    def fixture_payload(self,v,X,y):
        from morphtree.species import AlgorithmSpecies
        payload=copy.deepcopy(bound_reference(self.engine.q['input_root'],self.engine.q['software_templates'][v['task']]))
        p=v['params'];k=len(v['preprocessing']['encoded_classes']) if v['task']=='multiclass' else 1
        payload['booster']['trees']=payload['booster']['trees'][:p['n_estimators']*k]
        payload['booster']['steps']=[p['learning_rate']]*(p['n_estimators']*k)
        payload['report']={'backend':'python-stdlib','config':{'seed':v['seed']}}
        if v['task']=='binary':payload['algorithm']=AlgorithmSpecies(AlgorithmGenome(**v['structure']),HyperParams(**p),{'learning_recipe':payload['algorithm']['learning_recipe']|{'seed':v['seed']}}).to_dict()
        else:payload['parameters']=p
        scalar=p['n_estimators']*k;self.reserve(calls=1,trees=scalar)
        self.event('FIXTURE_FIT_INTENT',scalar_trees=scalar,key=digest(v),physical_fits=0)
        return payload

    def execute_native(self,model,X,y,*,identity,cancel):
        if evolution_profile():self._validate_evolution_input(model,identity)
        c=asdict(model.config);p=asdict(model.base_params);task=self.prepared['task'];family=identity['family'];i=identity['member']
        if c['seed']!=42+i:raise ValueError('Wrong member seed')
        ids=list(range(len(identity['actual_TRAIN_y']))) if family=='boost' else bootstrap(identity['actual_TRAIN_y'],task,42+i+1)
        if ids!=identity['ordered_sample']:raise ValueError('Changed actual bootstrap')
        if self.phase=='NUMERIC':
            if self.frozen is None or family!=self.frozen['family']:raise ValueError('Changed numeric family')
            g=model.base_genome.to_dict() if task=='binary' else AlgorithmGenome().to_dict()
            if g!=self.frozen['recipe']['genome']:raise ValueError('Changed frozen genome')
            base=HyperParams(**self.frozen['params']);base=base if family=='boost' else replace(base,n_estimators=1,learning_rate=1.,row_fraction=1.)
            allowed={'l2','min_samples_leaf'}|({'learning_rate'} if family=='boost' else set())
            if any(p[k]!=v for k,v in asdict(base).items() if k not in allowed):raise ValueError('Changed structural/fixed numeric field')
        return super().execute_native(model,X,y,identity=identity,cancel=cancel)

    def native_boundary(self,original,model,X,y,kwargs):
        self.check()
        if self.component is None:raise PermissionError('No active component lease')
        want,xx,yy,request=self.component;v=request.value()
        if model is not want or X!=xx or y!=yy or set(kwargs)!={'cancel'} or not callable(kwargs['cancel']):raise ValueError('Actual native invocation changed')
        if asdict(model.base_params)!=v['params'] or asdict(model.config)!=v['config']:raise ValueError('Actual native recipe changed')
        g=model.base_genome.to_dict() if v['task']=='binary' else AlgorithmGenome().to_dict()
        if g!=v['structure']:raise ValueError('Actual native structure changed')
        self.prepared.update(request=v,key=request.key)
        self.prefit+=1;self.event('NATIVE_PREFIT_BOUNDARY',key=request.key,physical_fits=0)
        if self.mode=='PREFIT':raise PreFitBoundary('STOP_BEFORE_ORIGINAL_NATIVE_FIT')
        if self.mode!='LIVE':raise PermissionError('Software cannot invoke original native function')
        self.permit();k=len(self.prepared['classes']) if v['task']=='multiclass' else 1
        scalar=v['params']['n_estimators']*k;self.reserve(calls=1,trees=scalar)
        self.event('NATIVE_FIT_STARTED',key=request.key,scalar_trees=scalar)
        self.physical+=1;self.native_started=time.monotonic()
        try:result=original(model,X,y,**kwargs);self.check()
        finally:self.native_started=None
        self.event('NATIVE_FIT_COMPLETED',key=request.key,scalar_trees=scalar)
        return result

    def permit(self):
        return validate_permit(self.owner,expected_owner_text_sha=self.owner_text_sha,engine=self.engine,
            prepared=self.campaign,policy=self.policy,workdir=self.root)

    def execute(self):
        if evolution_profile():
            if self.mode=='PREFIT' and self.policy['scenario']=='oof':
                from morphtree import MorphTree
                from .integrated import learning_program
                from .base import _payloads
                c=self.campaign['cases']['binary'];raw=c['raw']
                estimator=MorphTree(task='binary',random_state=42,verbosity=0,use_optuna=False,max_mutations=2,time_budget=300)
                self.bind_input(estimator,_NamedRows(raw['X'],raw['names']),raw['y'])
                r=self.engine.q['oof_parent'];bound_reference(self.root,r)
                retained=MorphTree.load(self.root/r['path'],workdir=self.root)
                p={'id':'retained-prefit-parent','task':'binary','family':retained.best_family_,
                   'recipe':retained.recipe()['structure'],'learning_identity':retained.recipe()['training_identity'],
                   'params':retained.recipe()['params'],'artifact_sha256':digest(_payloads(retained.model_))}
                with self.activate(),self.orchestration_operation(estimator):
                    self.evolution_context(p,learning_program(p))
                raise AssertionError('OOF PREFIT did not stop before original native learning')
            return live_pipeline(self)
        if self.mode!='PREFIT':return live_pipeline(self)
        from .integrated import MorphTree
        seen=[]
        for task,_,_,_ in WORKFLOWS[:3]:
            raw=self.campaign['cases'][task]['raw']
            model=MorphTree(task=task,families='boost',morphing='none',tuning='none',random_state=42,time_budget=300)
            try:
                with self.activate():model.fit(_NamedRows(raw['X'],raw['names']),raw['y'])
            except PreFitBoundary:seen.append(task)
            else:raise AssertionError('Original native fit reached in PREFIT')
        self.event('ALL_TASK_PREFIT_QUALIFIED',tasks=seen,physical_fits=0)
        raise PreFitBoundary('ALL_THREE_TASKS_BEFORE_ORIGINAL_NATIVE_FIT')

    def counters(self):
        counts=super().counters();used=self._used()
        rows=[json.loads(x) for x in (self.store.root/'EVENTS.jsonl').read_text('utf8').splitlines()] if (self.store.root/'EVENTS.jsonl').exists() else []
        counts.update(physical_fits_started=sum(r['event']=='NATIVE_FIT_STARTED' for r in rows),
            physical_fits_completed=sum(r['event']=='NATIVE_FIT_COMPLETED' for r in rows),
            Optuna_trials=sum(r['event']=='NUMERIC_TRIAL_INTENT' and r['real'] for r in rows),
            simulated_trials=sum(r['event']=='NUMERIC_TRIAL_INTENT' and not r['real'] for r in rows),
            stages=used,scalar_trees_started=sum(r.get('scalar_trees',0) for r in rows if r['event']=='NATIVE_FIT_STARTED'))
        counts['trials']=counts['Optuna_trials'];return counts


def live_pipeline(session):
    """Same installed API for every stage; no fit outside that API."""
    from .integrated import MorphTree
    if evolution_profile():
        from morphtree import MorphTree
    results=[]
    for index,(task,family,morph,tuning) in enumerate(WORKFLOWS):
        session.check();c=session.campaign['cases'][task];raw=c['raw']
        options=dict(task=task,tuning=tuning,n_trials=2,random_state=42,time_budget=300)
        model=(MorphTree(task=task,random_state=42,verbosity=0,use_optuna=False,max_mutations=2,time_budget=300)
               if evolution_profile() else MorphTree.research(operators=('growth_policy:depthwise',),**options) if morph=='auto'
               else MorphTree(families=family,morphing=morph,**options))
        with session.activate():model.fit(_NamedRows(raw['X'],raw['names']),raw['y'])
        folder=artifact_path(session.root/('result-'+str(index)));folder.mkdir()
        from .postfit import finish
        result=finish(session,model,folder,c['raw']);results.append(result)
        session.event('WORKFLOW_COMPLETED',index=index,task=task,model=result['model'],counts=session.counters())
        if hasattr(model,'program_history_'):
            from .base import _payloads
            def finalized(packet):
                if digest(_payloads(model.model_))!=packet['artifact_sha256']:
                    raise ValueError('Postfit weights differ from verified frozen packet')
                return {'packet_sha256':digest(packet),'weights_sha256':packet['artifact_sha256'],
                    'model':result['model'],'V3':result['V3'],'standalone':result['standalone'],
                    'predictor':result['program']}
            with session.activate():model._verify_evolution_report(finalization=finalized)
            result['report']=model.report()
    if session.cursor!=len(WORKFLOWS):raise ValueError('Missing planned workflows')
    return {**results[-1],'kind':'LIVE_NATIVE_MULTIPHASE' if session.mode=='LIVE' else 'SOFTWARE_MULTIPHASE_NOT_TRAINING',
        'workflows':results,'counts':session.counters(),'pipeline_status':'ALL_PLANNED_STAGES_COMPLETE',
        'recipe':model.recipe(),'report':model.report()}

def _original_claim_once(input_root, token, value):
    """Nonrenewable cooperative consumption across all output directories.

    A token is not permission. The separate trusted owner/protocol/identity
    checks are still mandatory before any LIVE operation.
    """
    if type(token) is not str or len(token) != 64 or any((c not in '0123456789abcdef' for c in token)):
        raise ValueError('Canonical consumption token')
    directory = safe_path(input_root, Path(input_root) / '.native-authorities' / token)
    with exclusive_lock(directory):
        if (directory / 'CLAIM.json').exists() or (directory / 'CLAIM.json.pending').exists():
            raise RuntimeError('Authority already consumed, across output roots')
        return atomic_result(directory, 'CLAIM.json', value, lambda: None)

def validate_permit(permit, *, expected_owner_text_sha, engine, prepared, policy, workdir):
    """A trusted caller supplies the owner's newly recorded message hash.

    A fixture, old receipt or request metadata is never itself permission.
    This cooperative trust boundary does not sandbox hostile Python callers.
    """
    if not expected_owner_text_sha:
        raise PermissionError('Separate explicit owner consent absent')
    if type(permit) is not dict or set(permit) != {'schema', 'scope', 'owner_text', 'runtime', 'protocol', 'plan_id', 'native_key', 'profile', 'output_root', 'phases', 'deadline_utc', 'policy_sha256'}:
        raise ValueError('Exact owner permit schema')
    if permit['schema'] != 'morph031-owner-fit-permit-v1' or permit['scope'] != 'EXPLICIT_OWNER_APPROVED_COMP009_EVOLUTION':
        raise PermissionError('Not a new executable owner permit')
    owner = bound_reference(engine.q['input_root'], permit['owner_text'])
    if permit['owner_text']['sha256'] != expected_owner_text_sha or owner.get('authority') != 'EXPLICIT_HUMAN_MESSAGE' or owner.get('campaign') != 'COMP009-LIVE-001':
        raise PermissionError('Unknown/consumed consent origin')
    p = prepared['plan']
    if owner.get('approved_profile')!=PROFILE or owner.get('approved_runtime')!=engine.identity or owner.get('approved_protocol')!=p['protocol'] or owner.get('authority_root')!=engine.q['authority_root']:raise PermissionError('Owner identity/profile/protocol mismatch')
    if p['protocol'].get('id') != 'COMP-009-LIVE-001' or not engine.q.get('training_protocol'):
        raise PermissionError('Zero-fit INT010 cannot authorize physical training')
    protocol = bound_reference(engine.q['input_root'], engine.q['training_protocol'])
    if protocol.get('id') != 'COMP-009-LIVE-001' or protocol.get('frozen_before_execution') is not True or p['protocol']['sha256'] != engine.q['training_protocol']['sha256'] or (protocol.get('planned_profile') != PROFILE):
        raise PermissionError('Not the separately frozen physical pilot protocol')
    if permit['runtime'] != engine.identity or permit['protocol'] != p['protocol'] or permit['plan_id'] != p['id'] or (permit['native_key'] != prepared['key']):
        raise ValueError('Consent bound to another runtime/request')
    if permit['profile'] != PROFILE or prepared['task'] != 'multiphase' or prepared['request']['learning_budget'] != {'calls': 13, 'scalar_trees': 14, 'trials': 0}:
        raise ValueError('Pilot exact fit/tree/task budget')
    if permit['output_root'] != str(Path(workdir).resolve()) or permit['phases'] != list(PHASES):
        raise ValueError('Wrong output/operation scope')
    activation = {k: v for k, v in policy.items() if k not in ('owner_permit', 'owner_text_sha256')}
    if permit['policy_sha256'] != digest(activation) or permit['deadline_utc'] != policy['deadline_utc'] or time.time() >= permit['deadline_utc']:
        raise PermissionError('Expired or different activation policy')
    claim = read(safe_path(engine.q['authority_root'], Path(engine.q['authority_root']) / '.native-authorities' / expected_owner_text_sha / 'CLAIM.json'))
    expected = {'owner_text_sha256': expected_owner_text_sha, 'native_key': prepared['key'], 'output_root': str(Path(workdir).resolve()), 'policy_sha256': digest(activation)}
    if claim != expected:
        raise PermissionError('Missing/different globally consumed authority')
    return True

def atomic_result(root, name, value, check):
    root = Path(root)
    target = safe_path(root, name)
    publish(target, json.dumps(value, sort_keys=True, allow_nan=False) + '\n', check)
    return {'path': target.relative_to(root).as_posix(), 'bytes': target.stat().st_size, 'sha256': sha(target)}

def claim_once(root,token,value):
    previous=WORKDIR.get();configure(root)
    try:return _original_claim_once(root,token,value)
    finally:WORKDIR.set(previous)
