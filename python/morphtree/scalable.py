"""Private columnar evolutionary facade, not a change to published 2.0.0.

Same native boosters, two genuine native families, current-parent OOF diagnosis,
two executable operators, verified SEARCH acceptance, final frozen-program
Optuna and explicit all-input refit. No source dataset JSON materialization.
"""
from array import array
from collections import Counter
import copy
from dataclasses import asdict,replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from .booster import BinaryHistogramBooster
from .config import AlgorithmGenome,HyperParams
from .persistence import decode_json_bytes,LoadLimits,decode_booster,decode_multiclass_booster,atomic_text,regular_file_size,read_regular_bytes
from .scalable_data import Input,Table,Encoder,Resources,scalar
from .scalable_backend import ScalableBackend
from .scalable_receipts import Ledger,verify as verify_receipts,data_identity,encoder_digest
from .prepared_bins import BinCache
from .resource_plan import fit_plan,memory_allowance,inference_seconds,io_seconds,memory_snapshot,numeric_capacity_plan
from .scalable_target import detect_target,label_scalar,compact_oof_train_indices
from .source_markers import controlled_source
from ._experimental.target import _split
from ._experimental.sampling import bootstrap,oof_folds

_PACKAGE=Path(__file__).resolve().parent
_ENGINE_SHA256=hashlib.sha256((_PACKAGE/'_experimental/manifest.json').read_bytes()).hexdigest()


def verify_source_identity():
    raw=(_PACKAGE/'_experimental/manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=_ENGINE_SHA256:raise ImportError('Changed source manifest since import')
    m=json.loads(raw)
    actual={p.relative_to(_PACKAGE).as_posix() for p in _PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=set(m['files'])|{'_experimental/manifest.json'}:raise ImportError('Missing/unexpected private runtime source')
    for n,h in m['files'].items():
        if hashlib.sha256((_PACKAGE/n).read_bytes()).hexdigest()!=h:raise ImportError('Changed private engine source')


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def point_loss(task,y,p):
    # Invalid internal predictions must stop selection, not become an apparently
    # excellent loss through NaN comparison/clipping or zip truncation.
    if task not in ('binary','multiclass','regression') or not len(y) or len(y)!=len(p):raise ValueError('Known task and nonempty aligned loss inputs required')
    if task=='regression':
        if any(not math.isfinite(a) or not math.isfinite(b) or not math.isfinite((a-b)**2) for a,b in zip(y,p)):raise ValueError('Finite regression loss inputs and squared errors required')
    elif task=='binary':
        if any(a not in (0,1) or not math.isfinite(b) or not 0<=b<=1 for a,b in zip(y,p)):raise ValueError('Binary labels and finite probabilities in [0,1] required')
    else:
        k=len(p[0])
        if k<2 or any(type(a) is not int or not 0<=a<k or len(b)!=k or any(not math.isfinite(v) or not 0<=v<=1 for v in b) or abs(math.fsum(b)-1)>1e-10 for a,b in zip(y,p)):raise ValueError('Aligned multiclass labels and finite probability simplexes required')
    if task=='regression':return array('d',((a-b)**2 for a,b in zip(y,p)))
    if task=='binary':return array('d',(-math.log(max(1e-15,min(1-1e-15,b if a else 1-b))) for a,b in zip(y,p)))
    return array('d',(-math.log(max(1e-15,min(1-1e-15,b[a]))) for a,b in zip(y,p)))
def risk(task,y,p):return statistics.mean(point_loss(task,y,p))
def available_memory():
    try:
        import os
        if os.name=='nt':
            import ctypes
            class Status(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            s=Status();s.length=ctypes.sizeof(s)
            if not ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('RAM query')
            return s.available
        return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,AttributeError):return 512*1024**2


class Family:
    def __init__(self,task,genome,params,encoder,models,family):self.task,self.genome,self.params,self.encoder,self.models,self.family=task,genome,params,encoder,models,family
    def probabilities(self,table,resources=None):
        if not self.models or not len(table):raise ValueError('Nonempty family and observations required')
        k=self.models[0].n_classes if self.task=='multiclass' else 1
        if resources is None:resources=managed_resources('auto',inference_seconds(len(table),table.width,sum(len(m.trees) for m in self.models),self.params.max_depth))
        output_bytes=len(table)*(64+40*k if self.task=='multiclass' else 40)+64
        member_row_bytes=64+40*k if self.task=='multiclass' else 40
        row_bytes=table.width*32+len(self.models)*member_row_bytes+256
        resources.reserve(output_bytes+row_bytes*4,'Family prediction output and member workspace')
        batch=max(1,min(16384,(resources.bytes_limit-output_bytes)//max(1,row_bytes*4)));result=[]
        for start in range(0,len(table),batch):
            resources.check();stop=min(len(table),start+batch);block=table if start==0 and stop==len(table) else table.take(range(start,stop));X=self.encoder.transform(block);parts=[]
            for model in self.models:
                resources.check();parts.append(model.predict_raw(X) if self.task=='regression' else model.predict_proba(X))
            # Preserve original member order and math.fsum for each observation.
            if self.task=='multiclass':result.extend([[math.fsum(v[i][j] for v in parts)/len(parts) for j in range(k)] for i in range(len(block))])
            else:result.extend([math.fsum(v[i] for v in parts)/len(parts) for i in range(len(block))])
            del parts,X,block
        resources.check();return result
    def packet(self):return dict(task=self.task,family=self.family,genome=self.genome.to_dict(),params=asdict(self.params),encoder=self.encoder.state(),members=[m.to_dict() for m in self.models])


def managed_resources(memory_budget,seconds,observer=None):
    observation=memory_snapshot(available_memory());observed=observation['effective_available_bytes'];res=Resources(memory_allowance(memory_budget,observed),seconds,observer)
    res.available_bytes_snapshot=observed;res.memory_observation=observation;res.base_commit=res.peak_commit=observation['process_private_bytes'];res.pressure_reader=lambda:memory_snapshot(available_memory());return res


def mutation_order(regions,guidance=True):
    """Research control changes only proposal order; OOF is still computed."""
    if type(guidance) is not bool:raise ValueError('Typed research guidance flag required')
    return ('split_program','growth_policy') if regions and guidance else ('growth_policy','split_program')


def streaming_loss_regions(table,point,folds,resources,primary=None):
    n=len(table)
    if not n or len(point)!=n:raise ValueError('Aligned nonempty OOF point losses')
    resources.reserve(n+table.width*256,'Compact diagnostic masks and output metadata')
    if primary is None:primary=statistics.mean(point)
    groups=[]
    def checked_column(feature):
        for index,value in enumerate(table.column(feature)):
            if index%4096==0:resources.check()
            yield value
    for feature,kind in enumerate(table.kinds):
        resources.check()
        if kind=='categorical':
            resources.reserve(n+table.width*256+min(n,len(table.vocab[feature])+1)*192,'Nominal diagnostic mode dictionary')
            cut=Counter(checked_column(feature)).most_common(1)[0][0]
            mask=bytearray(value==cut for value in checked_column(feature))
        else:
            try:cut=statistics.mean(value for value in checked_column(feature) if math.isfinite(value))
            except statistics.StatisticsError:cut=0
            mask=bytearray(math.isfinite(value) and value>cut for value in checked_column(feature))
        contrasts=[]
        for held in folds:
            resources.check();selected=sum(mask[index] for index in held)
            if min(selected,len(held)-selected)>=8:
                first=statistics.mean(point[index] for index in held if mask[index])
                second=statistics.mean(point[index] for index in held if not mask[index])
                contrasts.append(first-second)
        if len(contrasts)==2 and min(contrasts)>.02*primary+.01:
            groups.append(dict(feature=feature,fold_excess=contrasts))
    resources.check()
    return groups


_PARAMETERS_V1=('random_state','verbosity','time_budget','memory_budget','use_optuna','n_trials','max_mutations','refit_full','capacity','checkpoint_dir')
_PARAMETERS_V2=_PARAMETERS_V1+('n_jobs','mode','operators')
_OPERATORS={'split_program:objective_balance':'split_program','growth_policy:depthwise':'growth_policy'}


class _Scalable:
    task='auto'
    def __init__(self,random_state=None,verbosity=0,*,time_budget='auto',memory_budget='auto',use_optuna=True,n_trials='auto',max_mutations=2,refit_full=True,capacity=None,checkpoint_dir='auto',n_jobs=1,mode='adaptive',operators=None):
        if random_state is not None and type(random_state) is not int or type(use_optuna) is not bool or type(refit_full) is not bool:raise ValueError('Integer seed or None and boolean stage controls required')
        if type(max_mutations) is not int or max_mutations<0:raise ValueError('Nonnegative integer mutation budget required')
        if n_trials!='auto' and (type(n_trials) is not int or n_trials<0):raise ValueError('Nonnegative integer trial budget or auto required')
        if n_jobs!='auto' and (type(n_jobs) is not int or n_jobs!=1):raise ValueError('This serial native controller supports n_jobs=1 or auto')
        if mode not in ('adaptive','baseline','research'):raise ValueError('Known baseline/adaptive/research mode required')
        if mode=='baseline' and operators is not None:raise ValueError('Baseline has no structural operators')
        if mode=='research' and operators!=('growth_policy:depthwise',):raise ValueError('Research requires explicit growth_policy:depthwise')
        if operators is not None and (type(operators) is not tuple or not operators or any(type(v) is not str or v not in _OPERATORS for v in operators) or len(set(operators))!=len(operators)):raise ValueError('Distinct supported structural operators required')
        if time_budget!='auto' and (type(time_budget) not in (int,float) or not math.isfinite(time_budget) or time_budget<=0):raise ValueError('Positive finite time allowance or auto')
        if memory_budget!='auto' and (type(memory_budget) is not int or memory_budget<1):raise ValueError('Positive byte allowance or auto')
        if type(verbosity) is not int or not 0<=verbosity<=2 or capacity is not None and type(capacity) is not dict:raise ValueError('Typed verbosity/capacity')
        if capacity:replace(HyperParams(),**capacity)
        if checkpoint_dir is not None and (not isinstance(checkpoint_dir,(str,Path)) or isinstance(checkpoint_dir,str) and not checkpoint_dir):raise ValueError('Local checkpoint parent path, auto or None required')
        self.checkpoint_dir=checkpoint_dir
        self.random_state=random_state;self.verbosity=verbosity;self.time_budget=time_budget;self.memory_budget=memory_budget;self.use_optuna=use_optuna;self.n_trials=n_trials;self.max_mutations=max_mutations;self.refit_full=refit_full;self.capacity=capacity;self.n_jobs=n_jobs;self.mode=mode;self.operators=operators;self._observer=None
    def _resolved_controls(self):
        # Revalidate direct parameter edits before input access or allocation.
        type(self)(**self.get_params(deep=False))
        genes=() if self.mode=='baseline' else tuple(_OPERATORS[v] for v in self.operators) if self.operators is not None else tuple(_OPERATORS.values())
        # Each of the two qualified operators changes a gene to one fixed value.
        # Along the single retained lineage at most m+(m-1)+...+1 children
        # can be attempted; duplicate parent/program pairs are never retrained.
        mutations=min(self.max_mutations,len(genes)*(len(genes)+1)//2)
        trials=(2 if self.n_trials=='auto' else self.n_trials) if self.use_optuna else 0
        return dict(mutations=mutations,trials=trials,genes=genes,families=('boost',) if self.mode=='baseline' else ('boost','bag'),
                    automatic_trial_policy='Conservative qualified count 2; numeric capacity and deadlines adapt to resources',
                    mutation_scope='Maximum attempts under current finite monotone operator grammar; not requested accepted mutations')
    def __sklearn_tags__(self):
        # Optional integration only; importing/training MorphTree needs no sklearn.
        from sklearn.utils import Tags,TargetTags,InputTags,ClassifierTags,RegressorTags
        classifier=self._estimator_type=='classifier'
        return Tags(estimator_type=self._estimator_type,target_tags=TargetTags(required=True),
                    input_tags=InputTags(allow_nan=True,categorical=True,string=True),
                    classifier_tags=ClassifierTags() if classifier else None,
                    regressor_tags=None if classifier else RegressorTags())
    def __sklearn_is_fitted__(self):return hasattr(self,'model_')
    def as_sklearn(self):
        """Optional array-output view for sklearn scoring/Pipeline; same native estimator."""
        from .sklearn_view import SklearnView
        return SklearnView(self)
    def _resources(self):
        return managed_resources(self.memory_budget,150 if self.time_budget=='auto' else self.time_budget,self._observer)
    def _bin_cache_limit(self,res):return min(32*1024**2,res.bytes_limit//8)
    def fit(self,X,y):
        previous=dict(self.__dict__);start=time.perf_counter();self.last_attempt_=dict(status='IN_PROGRESS',native_fits=0,OOF_fits=0,scalar_trees=0,Optuna_trials=0,cache_hits=0,events=[])
        self._ledger=Ledger();self._attempt_anchor=None;self._active_checkpoint_dir_=None
        self.last_attempt_['requested_stages']=dict(max_mutations=self.max_mutations,use_optuna=self.use_optuna,Optuna_trials=self.n_trials if self.use_optuna else 0,refit_full=self.refit_full,diagnostic_guidance=getattr(self,'_research_use_diagnosis',True))
        try:
            verify_source_identity();self._event('CALL_STARTED',engine_sha256=_ENGINE_SHA256,scope='PRIVATE_QUALITY019_NOT_PUBLISHED',requested_task=self.task)
            type(self)(**self.get_params(deep=False))
            self._requested_random_state=self.random_state
            if self.random_state is None:
                import random
                self.random_state=random.SystemRandom().randrange(2**31)
            self.last_attempt_['effective_seed']=self.random_state
            self.last_attempt_['resolved_n_jobs']=1
            self._fit(X,y);self.__dict__.pop('_retained_variants',None);self.__dict__.pop('_variant_selection',None);self.last_attempt_['status']='COMPLETED';self.last_attempt_['seconds']=time.perf_counter()-start;self.report_=self.last_attempt_
            if self.last_attempt_['native_fits']:
                from .scalable_checkpoint import commit
                commit(self,self.model_.packet(),self.classes_,self._res,final=True)
            self.last_attempt_['seconds']=time.perf_counter()-start;self.__dict__.pop('_res',None)
            self.__dict__.pop('_interrupted_anchor',None);self.__dict__.pop('_interrupted_evidence',None)
            return self
        except BaseException as exc:
            anchor=self._attempt_anchor;evidence=getattr(self,'diagnostic_profiles_',{})
            attempt=self.last_attempt_;ledger=self._ledger
            # Restore fitted state before any fallible error reporting. A single
            # dictionary swap avoids exposing a cleared or partly restored state.
            self.__dict__=previous
            self.last_attempt_=attempt;self._interrupted_anchor=anchor;self._interrupted_evidence=evidence
            try:message=str(exc)
            except BaseException:message='<exception message unavailable>'
            attempt.update(status='STOPPED_NO_RETRY',error=type(exc).__name__,message=message,seconds=time.perf_counter()-start)
            if attempt['events']:
                try:
                    attempt['events'].append(ledger.emit('FIT_STOPPED',dict(error_type=type(exc).__name__)))
                    attempt['failure_receipt_status']='RECORDED_PARTIAL_CHAIN'
                except BaseException as receipt_error:
                    attempt['failure_receipt_status']='TERMINAL_EVENT_UNAVAILABLE'
                    attempt['failure_receipt_error']=type(receipt_error).__name__
            else:attempt['failure_receipt_status']='CALL_CHAIN_NOT_STARTED'
            raise
        finally:
            self.random_state=previous['random_state']
            self.__dict__.pop('_requested_random_state',None)
    def _event(self,kind,**kw):self.last_attempt_['events'].append(self._ledger.emit(kind,kw))
    def _ordinary_context_guard(self):
        # An active or orphaned scientific/quota context must never fall through
        # to ordinary memory data. The legacy scientific API owns those contexts.
        from .api import ACTIVE, PUBLIC_SCOPE
        from ._experimental.admission import CURRENT, AdmissionError
        from ._experimental.producer import PIPE_CURRENT
        if any(value.get() is not None for value in (ACTIVE,CURRENT,PIPE_CURRENT,PUBLIC_SCOPE)):
            raise AdmissionError({'status':'REJECTED','reason':'CONTROLLED_EXECUTION_CONTEXT_CANNOT_FALL_THROUGH_TO_ORDINARY_DATA'})

    def _fit(self,X,y):
        self._ordinary_context_guard()
        from .native_lab_backend import bind_backend
        backend_factory=bind_backend()
        if any(controlled_source(obj) for obj in (X,y)):raise ValueError('Controlled sources cannot use this ordinary experimental input path')
        controls=self._resolved_controls();mutations=controls['mutations'];trials=controls['trials'];self.last_attempt_['resolved_controls']=controls
        res=self._resources();self._res=res;tick=time.perf_counter();source=Input(X)
        hp=replace(HyperParams(n_estimators=2,learning_rate=1.,max_leaves=5,max_depth=3,max_bins=15,min_samples_leaf=4,row_fraction=1.,feature_fraction=1.),**(self.capacity or {}))
        available=res.available_bytes_snapshot
        plan=fit_plan(source.n,source.width,1 if self.task=='regression' else 2,hp,mutations,trials,self.refit_full,available,self.time_budget,res.bytes_limit)
        res.configure_time(plan['seconds']);self.last_attempt_['resource_plan']=plan
        res.reserve(source.n*source.width*10+source.n*(256+80*(1 if self.task=='regression' else 2))+source.width*512,'Minimum configured fit workspace before target/feature allocation')
        task,targets,classes=detect_target(y,self.task,res.check,reserve=res.reserve)
        if len(targets)!=source.n:raise ValueError('Aligned X/y required')
        k=len(classes) if task!='regression' else 1
        plan=fit_plan(source.n,source.width,k,hp,mutations,trials,self.refit_full,available,self.time_budget,res.bytes_limit)
        plan=numeric_capacity_plan(plan,hp,mutations,trials,self.refit_full,'n_estimators' in (self.capacity or {}))
        plan['memory_mode']='auto' if self.memory_budget=='auto' else 'explicit'
        plan['memory_observation']=res.memory_observation
        res.configure_time(plan['seconds']);self.last_attempt_['resource_plan']=plan;self._event('RESOURCE_PLAN',plan=plan)
        res.reserve(plan['estimated_workspace_bytes'],'Conservative fit workspace before feature allocation')
        raw=Table.read(source,res)
        self._event('INPUT_PREPARED',rows=len(raw),features=raw.width,seconds=time.perf_counter()-tick,storage_bytes=raw.storage_bytes,input_identity=data_identity(raw,targets,'USER_INPUT'))
        ti,si=_split(targets,task,self.random_state)
        train,search=raw.take(ti),raw.take(si);ty=array(targets.typecode,(targets[i] for i in ti));sy=array(targets.typecode,(targets[i] for i in si));del ti,si
        cache={};prepared=BinCache(self._bin_cache_limit(res),res.check);task_classes=k
        def acquire(family,table,labels,genome,params,phase,seed):
            res.check();enc=Encoder().fit(table);encoded=enc.transform(table)
            identity=data_identity(table,labels,'TRAIN_FULL' if phase=='REFIT_FULL' else 'TRAIN');preprocessing=encoder_digest(enc)
            key=digest(dict(data=identity,preprocessing_sha256=preprocessing,task=task,family=family,genome=genome.to_dict(),params=asdict(params),seed=seed,engine_sha256=_ENGINE_SHA256))
            if key in cache:
                self.last_attempt_['cache_hits']+=1;self._event('CACHE_HIT',phase=phase,family=family,rows=len(table),cache_key=key);return cache[key]
            models=[];m=1 if family=='boost' else params.n_estimators
            for member in range(m):
                res.check();actual=params if family=='boost' else replace(params,n_estimators=1,learning_rate=1.)
                if family=='bag':
                    ids=bootstrap(labels,task,seed+member+1);t=table.take(ids);yy=array(labels.typecode,(labels[i] for i in ids));del ids;xx=enc.transform(t)
                else:xx,yy=encoded,labels
                self.last_attempt_['native_fits']+=1;self.last_attempt_['OOF_fits']+=phase=='OOF';self.last_attempt_['scalar_trees']+=actual.n_estimators*(k if task=='multiclass' else 1)
                actual_table=xx.source;effective_rows=hashlib.sha256(memoryview(actual_table.indices)).hexdigest() if actual_table.indices is not None else digest(dict(range_start=0,range_stop=len(actual_table)))
                effective_labels=hashlib.sha256(memoryview(yy)).hexdigest();component=digest(dict(request=key,member=member,effective_rows=effective_rows,effective_labels=effective_labels,seed=seed+member,params=asdict(actual)))
                self._event('NATIVE_FIT_STARTED',ordinal=self.last_attempt_['native_fits'],phase=phase,rows=len(yy),source_rows=len(table),family=family,cache_key=key,component_key=component,effective_rows_sha256=effective_rows,effective_labels_sha256=effective_labels,seed=seed+member,params_sha256=digest(asdict(actual)),preprocessing_sha256=preprocessing,program_sha256=digest(genome.to_dict()),role=identity['role'],reserved_scalar_trees=actual.n_estimators*(k if task=='multiclass' else 1))
                res.check()
                model=BinaryHistogramBooster(genome,actual,seed+member,backend_factory(encoded._nominal_sizes),memory_enabled=False,objective='softmax' if task=='multiclass' else 'squared_error' if task=='regression' else 'binary_logloss',n_classes=k if task=='multiclass' else None)
                preparation_identity=dict(source_sha256=identity['source_sha256'],schema_sha256=identity['schema_sha256'],effective_rows_sha256=effective_rows,effective_labels_sha256=effective_labels,preprocessing_sha256=preprocessing,engine_sha256=_ENGINE_SHA256)
                token=prepared.bind(xx,preparation_identity,'TRAIN_FULL' if phase=='REFIT_FULL' else 'OOF_TRAIN' if phase=='OOF' else 'TRAIN')
                tick=time.perf_counter();model.fit(xx,yy,callback=lambda *_:res.check() or True,prepared_data=token)
                assert len(model.trees)==actual.n_estimators*(k if task=='multiclass' else 1)
                self._event('NATIVE_FIT_COMPLETED',ordinal=self.last_attempt_['native_fits'],phase=phase,seconds=time.perf_counter()-tick,component_key=component,actual_scalar_trees=len(model.trees));models.append(model)
            fitted=Family(task,genome,params,enc,models,family);cache[key]=fitted;return fitted
        base=AlgorithmGenome();tick=time.perf_counter();screen={f:acquire(f,train,ty,base,hp,'SCREEN',self.random_state) for f in controls['families']}
        losses={f:risk(task,sy,m.probabilities(search,res)) for f,m in screen.items()};family=min(losses,key=lambda f:(losses[f],f));current=screen[family];current_risk=losses[family]
        self._event('SCREEN_COMPLETED',losses=losses,parent=family,seconds=time.perf_counter()-tick)
        self._research_initial_parent=current
        def retain_anchor():
            self._attempt_anchor=dict(model=current.packet(),classes=classes,SEARCH=current_risk,source_role='TRAIN',final_refit=False)
            self.last_attempt_['last_verified_anchor']=dict(model_sha256=digest(current.packet()),SEARCH=current_risk,source_role='TRAIN',final_refit=False)
            from .scalable_checkpoint import commit
            commit(self,self._attempt_anchor['model'],classes,res,final=False)
        retain_anchor()
        seen=set();diagnoses={};self.diagnostic_profiles_=diagnoses
        for generation in range(mutations):
            parent=digest(current.packet())
            if parent not in diagnoses:
                tick=time.perf_counter();folds=oof_folds(ty,task,self.random_state);oof=[None]*len(ty)
                for held in folds:
                    ids=compact_oof_train_indices(len(ty),held,res);fold=acquire(family,train.take(ids),array(ty.typecode,(ty[i] for i in ids)),current.genome,hp,'OOF',self.random_state)
                    pred=fold.probabilities(train.take(held),res)
                    self._event('OOF_FOLD',parent_sha256=parent,program_sha256=digest(current.genome.to_dict()),fit_rows=len(ids),held_rows=len(held),disjoint=True,encoder_sha256=digest(fold.encoder.state()),fit_indices_sha256=hashlib.sha256(array('Q',ids)).hexdigest(),held_indices_sha256=hashlib.sha256(array('Q',held)).hexdigest())
                    for i,p in zip(held,pred):oof[i]=p
                point=point_loss(task,ty,oof);primary=statistics.mean(point)
                groups=streaming_loss_regions(train,point,folds,res,primary)
                diagnoses[parent]=dict(parent_sha256=parent,task=task,OOF_primary=primary,fold_rows=list(map(len,folds)),TRAIN_only=True,fold_encoder_local=True,features_examined=train.width,regions=groups,causal_status='HEURISTIC_UNCONFIRMED',seconds=time.perf_counter()-tick)
                profile=diagnoses[parent]
                self._event('DIAGNOSIS',**{key:value for key,value in profile.items() if key!='regions'},region_count=len(groups),profile_sha256=digest(profile));del oof,point,folds
            diag=diagnoses[parent];order=mutation_order(diag['regions'],getattr(self,'_research_use_diagnosis',True));proposal=None
            for gene in order:
                if gene not in controls['genes']:continue
                value=('OBJECTIVE_GAIN','BALANCE','MUL') if gene=='split_program' else 'depthwise'
                if getattr(current.genome,gene)==value:continue
                candidate=current.genome.mutate(**{gene:value});identity=digest(candidate.to_dict())
                if (parent,identity) not in seen:proposal=candidate;seen.add((parent,identity));break
            if proposal is None:break
            tick=time.perf_counter();child=acquire(family,train,ty,proposal,hp,'DESCENDANT',self.random_state);child_risk=risk(task,sy,child.probabilities(search,res));accept=child_risk<current_risk-1e-4*max(1.,abs(current_risk))
            self._event('MUTATION',generation=generation,parent_sha256=parent,child_sha256=digest(child.packet()),child_program=proposal.to_dict(),parent_SEARCH=current_risk,child_SEARCH=child_risk,decision='ACCEPT' if accept else 'ROLLBACK',seconds=time.perf_counter()-tick)
            if accept:current=child;current_risk=child_risk;retain_anchor()
        frozen=current.genome;self._event('PROGRAM_FROZEN',genome_sha256=digest(frozen.to_dict()))
        self._research_structural_before_numeric=current
        if self.use_optuna:
            import optuna
            maximum_estimators=plan['numeric_capacity']['maximum_estimators']
            study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=self.random_state,n_startup_trials=2))
            for _ in range(trials):
                res.check();trial=study.ask();self.last_attempt_['Optuna_trials']+=1
                p=replace(hp,l2=trial.suggest_float('l2',.1,10.,log=True),min_samples_leaf=trial.suggest_int('min_samples_leaf',2,8))
                if maximum_estimators>hp.n_estimators:p=replace(p,n_estimators=trial.suggest_int('n_estimators',hp.n_estimators,maximum_estimators))
                self._event('OPTUNA_ASK',number=trial.number,genome_sha256=digest(frozen.to_dict()))
                child=acquire(family,train,ty,frozen,p,'NUMERIC',self.random_state);value=risk(task,sy,child.probabilities(search,res));study.tell(trial,value)
                if value<current_risk:current=child;current_risk=value;retain_anchor()
                self._event('OPTUNA_COMPLETE',number=trial.number,SEARCH=value,parameters=asdict(p))
        self._research_selected_before_refit=current
        if self.refit_full:current=acquire(family,raw,targets,frozen,current.params,'REFIT_FULL',self.random_state)
        self.model_=current;self.classes_=classes;self.task_=task;self.last_attempt_['bin_cache']=prepared.report();self._event('BIN_PREPARATION_CACHE',**prepared.report());prepared.clear()
        self._event('FINAL_MODEL',input_rows=len(raw),selection_TRAIN_rows=len(train),SEARCH_rows=len(search),final_fit_source_rows=len(raw) if self.refit_full else len(train),refit_full=self.refit_full,SEARCH_primary_before_refit=current_risk,selection_metric_is_independent=False,nominal_columns=len(current.encoder.transform(raw)._nominal_sizes),source_storage_bytes=raw.storage_bytes,all_rows_accounted=True)
        self.n_features_in_=raw.width;self.best_family_=family;self.best_params_=asdict(current.params)
        self.last_attempt_['receipt_verification']=verify_receipts(self.last_attempt_['events'])
        self.last_attempt_['resource_usage']=dict(baseline_RSS_bytes=res.base_rss,sampled_peak_RSS_bytes=res.peak_rss,baseline_private_commit_bytes=res.base_commit,sampled_peak_private_commit_bytes=res.peak_commit)
        self.last_attempt_['scope']='PRIVATE_EXPERIMENTAL_NOT_PUBLISHED_2_0_OR_GENERAL_UTILITY_QUALIFICATION';cache.clear();gc.collect()
    def _blocks(self,X):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        res=self._resources();source=Input(X);k=len(self.classes_) if self.task_!='regression' else 1
        schema=self.model_.encoder
        if schema.names is None:
            if source.names is not None:raise ValueError('Positional TRAIN requires positional prediction')
            ordered=schema.columns
        else:
            if source.names is None or set(source.names)!=set(schema.names):raise ValueError('Feature schema mismatch')
            columns=dict(zip(schema.names,schema.columns));ordered=[columns[name] for name in source.names]
        if len(ordered)!=source.width:raise ValueError('Feature count mismatch')
        # Infer representations from the fitted schema,never refit vocabularies
        # from query dtypes. Preserve typed numeric nominal IDs through Table.
        source.fitted_kinds=tuple(column['kind'] for column in ordered)
        if self.time_budget=='auto':res.configure_time(inference_seconds(source.n,source.width,sum(len(m.trees) for m in self.model_.models),self.model_.params.max_depth))
        member_rows=len(self.model_.models)*(40*k+64 if self.task_=='multiclass' else 40)
        batch=max(1,min(16384,res.bytes_limit//max(1,(source.width*32+member_rows+k*128+256)*4)))
        for start in range(0,source.n,batch):
            raw=Table.read(source,res,start,min(source.n,start+batch));yield self.model_.probabilities(raw,res)
    def predict(self,X):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        import numpy as np
        from .prediction_output import prediction_array
        n=Input(X).n;res=self._resources()
        res.reserve(n*40+len(self.classes_)*64,'Returned prediction array and class dtype')
        dtype=np.float64 if self.task_=='regression' else prediction_array(self.classes_).dtype
        result=np.empty(n,dtype=dtype);offset=0
        for values in self._blocks(X):
            res.check();stop=offset+len(values)
            if stop>n:raise ValueError('Unexpected prediction row count')
            if self.task_=='regression':result[offset:stop]=values
            elif self.task_=='binary':result[offset:stop]=[self.classes_[int(p>=.5)] for p in values]
            else:result[offset:stop]=[self.classes_[max(range(len(p)),key=p.__getitem__)] for p in values]
            offset=stop
        if offset!=n:raise ValueError('Incomplete prediction rows')
        if self.task_=='regression' and not np.isfinite(result).all():raise ValueError('Finite regression predictions required')
        res.check()
        return result
    def predict_proba(self,X):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        if self.task_=='regression':raise AttributeError('Regression has no class probabilities')
        import numpy as np
        from .prediction_output import probability_array
        n=Input(X).n;k=len(self.classes_);res=self._resources();res.reserve(n*(k*40+64),'Returned probability array and bounded validation')
        result=np.empty((n,k),dtype=np.float64);offset=0
        for values in self._blocks(X):
            res.check();stop=offset+len(values)
            if stop>n:raise ValueError('Unexpected probability row count')
            result[offset:stop]=[[1-p,p] for p in values] if self.task_=='binary' else values
            offset=stop
        if offset!=n:raise ValueError('Incomplete probability rows')
        return probability_array(result,k,n_samples=n,check=res.check)
    def available_variants(self):
        from .scalable_variants import describe
        return describe(self)
    def _variant_view(self,variant):
        from .scalable_variants import view
        return view(self,variant)
    def save(self,path,*,variant='all'):
        from .scalable_variants import save_bundle
        result=save_bundle(self,path,variant)
        if result is not None:return result
        return self._save_single(path)
    def _save_single(self,path):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        res=self._serialization_resources('Model snapshot')
        chained=bool(self.report_.get('events')) and all('receipt_schema' in e for e in self.report_['events'])
        if chained:verify_receipts(self.report_['events'])
        packet=dict(schema='SCALABLE_PRIVATE_MODEL_2' if chained else 'SCALABLE_PRIVATE_MODEL_1',model=self.model_.packet(),classes=self.classes_,report=self.report_)
        if chained:packet['receipt_sha256']=digest(self.report_);packet['evidence']=dict(diagnostic_profiles=getattr(self,'diagnostic_profiles_',{}))
        if chained:
            packet['schema']='SCALABLE_PRIVATE_MODEL_3'
            packet['constructor']=self._constructor_state()
            packet['constructor_sha256']=digest(packet['constructor'])
        raw=json.dumps(packet,sort_keys=True,allow_nan=False)
        res.reserve(len(raw.encode())*4,'Serialized model');atomic_text(Path(path),raw+'\n');return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def _serialization_resources(self,description,*,families=None):
        res=self._resources()
        families=list({id(f):f for f in (families or [self.model_])}.values())
        members=[m for family in families for m in family.models]
        nodes=sum(len(tree.nodes) for model in members for tree in model.trees)
        cuts=sum(len(c) for model in members for c in model.binner.thresholds)
        words=sum(len(v) for family in families for c in family.encoder.columns for v in c.get('vocabulary',()))
        names=sum(len(v) for family in families for v in family.encoder.names or ())
        labels=sum(len(v) for v in self.classes_ if isinstance(v,str))
        estimate=1024+nodes*4096+cuts*64+(words+names+labels)*32+len(self.report_.get('events',()))*4096
        if self.time_budget=='auto':res.configure_time(io_seconds(estimate))
        res.reserve(estimate,description+' estimated serialization workspace');return res
    def export_python(self,path,*,variant='selected',output_format='python'):
        if output_format not in ('python','numpy'):raise ValueError('output_format must be python or numpy')
        if variant!='selected':return self._variant_view(variant).export_python(path,output_format=output_format)
        import inspect
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        res=self._serialization_resources('Standalone snapshot')
        packet=dict(model=self.model_.packet(),classes=self.classes_)
        guard_source=(_PACKAGE/'source_markers.py').read_text('utf8')
        source='# Private frozen Python inference; no training or external libraries.\nMODEL='+repr(packet)+'\n'+guard_source+'\n'+inspect.getsource(memory_snapshot)+'\n'+inspect.getsource(memory_allowance)+'\n'+inspect.getsource(available_memory)+'\n'+inspect.getsource(_standalone)+'\ndef predict(X):return _standalone(X,MODEL,False)\ndef predict_proba(X):return _standalone(X,MODEL,True)\n'
        if output_format=='numpy':
            from .prediction_output import numpy_export_source
            source=numpy_export_source(source,self.classes_,self.task_=='regression')
        res.reserve(len(source.encode())*4,'Standalone source');atomic_text(Path(path),source);return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @classmethod
    def load(cls,path,*,sha256=None,memory_budget='auto',workdir=None):
        from .api_compat import validate_workdir,wrap_historical
        workdir=validate_workdir(workdir)
        p=Path(path);size=regular_file_size(p);res=managed_resources(memory_budget,io_seconds(size));res.reserve(size*8,'Validated JSON model')
        limit=max(1,res.bytes_limit//8)
        raw=read_regular_bytes(p,limit)
        res.reserve(len(raw)*8,'Immutable JSON snapshot')
        actual_sha256=hashlib.sha256(raw).hexdigest()
        if sha256 is not None and actual_sha256!=sha256:raise ValueError('Changed saved model')
        packet=decode_json_bytes(raw,LoadLimits(max_bytes=limit),reserve=res.reserve)
        if type(packet) is not dict:raise ValueError('Typed model envelope must be an object')
        from .legacy_artifacts import PUBLIC_2_0_SCHEMA,restore_public_2_0
        if packet.get('schema')==PUBLIC_2_0_SCHEMA:
            if set(packet)!={'constructor','origins','schema','selected_variant','variants'} or type(packet['constructor']) is not dict or type(packet['origins']) is not dict or type(packet['selected_variant']) is not str or type(packet['variants']) is not dict or not packet['variants']:raise ValueError('Exact historical model envelope required')
            del packet,raw
            restored=restore_public_2_0(p,actual_sha256,cls.task,workdir=workdir);res.check()
            return wrap_historical(restored,cls,memory_budget)
        if packet.get('schema')=='SCALABLE_PRIVATE_VARIANTS_1':
            from .scalable_variants import restore_bundle
            return restore_bundle(cls,packet,res,limit,actual_sha256,sha256 is not None,memory_budget)
        return cls._restore_packet(packet,res,limit,actual_sha256,sha256 is not None,memory_budget)
    @classmethod
    def _restore_packet(cls,packet,res,limit,actual_sha256,expected_checksum_supplied,memory_budget):
        required={'schema','model','classes','report'}
        if packet.get('schema') in ('SCALABLE_PRIVATE_MODEL_2','SCALABLE_PRIVATE_MODEL_3'):required.update(('receipt_sha256','evidence'))
        if packet.get('schema')=='SCALABLE_PRIVATE_MODEL_3':required.update(('constructor','constructor_sha256'))
        if set(packet)!=required or packet['schema'] not in ('SCALABLE_PRIVATE_MODEL_1','SCALABLE_PRIVATE_MODEL_2','SCALABLE_PRIVATE_MODEL_3') or type(packet['report']) is not dict:raise ValueError('Private model schema')
        if packet['schema'] in ('SCALABLE_PRIVATE_MODEL_2','SCALABLE_PRIVATE_MODEL_3'):
            if packet['receipt_sha256']!=digest(packet['report']):raise ValueError('Changed report identity')
            verification=verify_receipts(packet['report'].get('events'))
            report=packet['report'];events=report['events'];starts=[e for e in events if e['event']=='NATIVE_FIT_STARTED'];ends=[e for e in events if e['event']=='NATIVE_FIT_COMPLETED']
            if any(type(e['ordinal']) is not int for e in starts) or [e['ordinal'] for e in starts]!=list(range(1,len(starts)+1)):raise ValueError('Invalid native component order')
            if report.get('status')=='COMPLETED' and (events[-1]['event']!='FINAL_MODEL' or not starts):raise ValueError('Completed training requires fitted-component and finalization receipts')
            if len(starts)!=report.get('native_fits') or sum(e['phase']=='OOF' for e in starts)!=report.get('OOF_fits') or sum(e['reserved_scalar_trees'] for e in starts)!=report.get('scalar_trees') or sum(e['event']=='OPTUNA_ASK' for e in events)!=report.get('Optuna_trials'):raise ValueError('Receipt/counter mismatch')
            if report.get('status')=='COMPLETED' and (len(starts)!=len(ends) or [e['component_key'] for e in starts]!=[e['component_key'] for e in ends] or sum(e['actual_scalar_trees'] for e in ends)!=report.get('scalar_trees')):raise ValueError('Incomplete fitted-component chain')
            evidence=packet['evidence']
            if type(evidence) is not dict or set(evidence)!={'diagnostic_profiles'} or type(evidence['diagnostic_profiles']) is not dict:raise ValueError('Separate diagnostic evidence schema')
            profiles=evidence['diagnostic_profiles'];receipts=[e for e in events if e['event']=='DIAGNOSIS']
            if set(profiles)!={e['parent_sha256'] for e in receipts} or any(digest(profiles[e['parent_sha256']])!=e['profile_sha256'] for e in receipts):raise ValueError('Changed diagnostic artifact identity')
        else:verification=dict(status='LEGACY_UNCHAINED_EXPERIMENTAL_REPORT',authority='Caller-supplied file custody only; not a verified operation chain')
        v=packet['model']
        if type(v) is not dict or set(v)!={'task','family','genome','params','encoder','members'} or v['task'] not in ('binary','multiclass','regression') or v['family'] not in ('boost','bag'):raise ValueError('Exact native model fields')
        if cls.task=='classification' and v['task']=='regression' or cls.task=='regression' and v['task']!='regression':raise ValueError('Saved task is incompatible with requested facade')
        facade=MorphTreeRegressor if v['task']=='regression' else MorphTreeClassifier
        if packet['schema']=='SCALABLE_PRIVATE_MODEL_3':
            state=packet['constructor']
            if digest(state)!=packet['constructor_sha256']:raise ValueError('Changed refit configuration identity')
            if type(state) is not dict or set(state)!={'schema','parameters','checkpoint_location_reset','scope'} or state['schema'] not in ('SCALABLE_REFIT_CONFIGURATION_1','SCALABLE_REFIT_CONFIGURATION_2') or type(state['parameters']) is not dict or type(state['checkpoint_location_reset']) is not bool or state['scope']!='FUTURE_REFIT_SETTINGS_AT_SNAPSHOT':raise ValueError('Exact typed refit configuration required')
            params=state['parameters']
            keys=_PARAMETERS_V1 if state['schema']=='SCALABLE_REFIT_CONFIGURATION_1' else _PARAMETERS_V2
            if set(params)!=set(keys) or params['checkpoint_dir'] not in ('auto',None):raise ValueError('Safe refit parameters required;artifact output paths cannot be restored')
            params=dict(params)
            if state['schema']=='SCALABLE_REFIT_CONFIGURATION_1':
                # Exact old contract; new constructor fields do not invalidate v1.
                if any(type(params[k]) is not int or not 0<=params[k]<=2 for k in ('n_trials','max_mutations')):raise ValueError('Invalid legacy refit configuration limits')
                params.update(n_jobs=1,mode='adaptive',operators=None)
            elif params['operators'] is not None:
                if type(params['operators']) is not list:raise ValueError('Serialized operator list required')
                params['operators']=tuple(params['operators'])
            if memory_budget!='auto':params['memory_budget']=memory_budget
            obj=facade(**params)
            recovery=dict(available=True,scope=state['scope'],checkpoint_location_reset=state['checkpoint_location_reset'],original_training_settings_inferred=False)
        else:
            obj=facade(memory_budget=memory_budget)
            recovery=dict(available=False,scope='LEGACY_CONFIG_NOT_RECORDED_DEFAULTS_FOR_FUTURE_REFIT',original_training_settings_inferred=False)
        from .native_lab_backend import bind_backend
        enc=Encoder.from_state(v['encoder'],res);hp=HyperParams(**v['params']);genome=AlgorithmGenome(**v['genome']);nominal={i:len(c['vocabulary']) for i,c in enumerate(enc.columns) if c['kind']=='categorical'};backend=bind_backend()(nominal);models=[]
        classes=packet['classes']
        if type(classes) is not list:raise ValueError('Typed class-label list required')
        if v['task']=='regression':
            if classes:raise ValueError('Regression class labels must be empty')
        else:
            normalized=[label_scalar(c) for c in classes]
            if any(c is None for c in normalized) or not (all(type(c) is str for c in normalized) or all(type(c) in (int,float) for c in normalized)):raise ValueError('Consistent finite scalar class labels required')
            if normalized!=sorted(set(normalized)) or len(classes)<2 or v['task']=='binary' and len(classes)!=2:raise ValueError('Distinct canonical task-compatible class labels required')
        if type(v['members']) is not list or len(v['members'])!=(1 if v['family']=='boost' else hp.n_estimators):raise ValueError('Exact bounded family members')
        for raw in v['members']:
            res.check()
            actual=hp if v['family']=='boost' else replace(hp,n_estimators=1,learning_rate=1.)
            limits=LoadLimits(max_features=len(enc.columns),max_trees=actual.n_estimators*(len(classes) if v['task']=='multiclass' else 1),max_nodes=limit,max_thresholds=limit)
            model=decode_multiclass_booster(raw,actual,backend,len(classes),'softmax',limits,genome=genome) if v['task']=='multiclass' else decode_booster(raw,genome,actual,backend,limits,objective='squared_error' if v['task']=='regression' else 'binary_logloss')
            if model.n_features_!=len(enc.columns):raise ValueError('Model/encoder feature mismatch')
            models.append(model)
            res.check()
        obj.model_=Family(v['task'],genome,hp,enc,models,v['family']);obj.classes_=classes;obj.task_=v['task'];obj.report_=copy.deepcopy(packet['report']);obj.report_['receipt_verification']=verification
        obj.report_['constructor_recovery']=recovery
        obj.report_['file_custody']=dict(sha256=actual_sha256,expected_checksum_supplied=expected_checksum_supplied,scope='A local computed hash alone does not authenticate an author or scientific claim')
        obj.diagnostic_profiles_=copy.deepcopy(packet.get('evidence',{}).get('diagnostic_profiles',{}))
        anchor=obj.report_.get('last_verified_anchor')
        if type(anchor) is dict and 'model' in anchor:obj.report_['last_verified_anchor']={k:value for k,value in anchor.items() if k not in ('model','classes')}|dict(model_sha256=digest(anchor['model']),legacy_embedded_model_omitted_from_report_view=True)
        obj.n_features_in_=len(enc.columns);obj.best_family_=v['family'];obj.best_params_=asdict(hp);return obj
    @classmethod
    def load_checkpoint(cls,directory,*,memory_budget='auto',stage='current'):
        """Recover current or selected TRAIN state;never resume training or auto-select from EXTERNAL."""
        from .scalable_checkpoint import read_pointer
        target,record=read_pointer(directory,managed_resources(memory_budget,150),stage=stage)
        obj=cls.load(target,sha256=record['sha256'],memory_budget=memory_budget);events=obj.report_.get('events',[])
        if not events or events[0]['call_id']!=record['call_id']:raise ValueError('Checkpoint call identity mismatch')
        if record['final']:
            final=events[-1]
            if obj.report_.get('status')!='COMPLETED' or final['event']!='FINAL_MODEL' or record['source_role']!=('TRAIN_FULL' if final['refit_full'] else 'TRAIN'):raise ValueError('Checkpoint finalization/role mismatch')
        elif obj.report_.get('status')!='PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT' or digest(obj.model_.packet())!=obj.report_.get('last_verified_anchor',{}).get('model_sha256'):raise ValueError('Checkpoint TRAIN anchor mismatch')
        obj.report_['checkpoint_recovery']=dict(**record,training_resumed=False)
        return obj
    def get_params(self,deep=True):
        params={k:getattr(self,k) for k in _PARAMETERS_V2}
        return copy.deepcopy(params) if deep else params
    def _constructor_state(self):
        params=self.get_params()
        if '_requested_random_state' in self.__dict__:params['random_state']=self._requested_random_state
        reset=params['checkpoint_dir'] not in ('auto',None)
        if reset:params['checkpoint_dir']='auto'
        # Validate executable future settings; never deserialize artifact-chosen
        # output destinations or present inferred defaults as old training facts.
        type(self)(**params)
        return dict(schema='SCALABLE_REFIT_CONFIGURATION_2',parameters=params,
                    checkpoint_location_reset=reset,scope='FUTURE_REFIT_SETTINGS_AT_SNAPSHOT')
    def set_params(self,**values):
        if not values:return self
        params=self.get_params(deep=False)
        if set(values)-set(params):raise ValueError('Unknown parameter')
        replacement=type(self)(**(params|values));self.__dict__.clear();self.__dict__.update(replacement.__dict__);return self
    def get_algorithm_program(self,*,variant='selected'):
        if variant!='selected':return self._variant_view(variant).get_algorithm_program()
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        frozen=any(e['event']=='PROGRAM_FROZEN' and e['genome_sha256']==digest(self.model_.genome.to_dict()) for e in self.report_.get('events',[]))
        return dict(schema='SCALABLE_EVOLVED_PROGRAM_1',task=self.task_,family=self.model_.family,genome=self.model_.genome.to_dict(),frozen_before_numeric=frozen,nominal_split='ONE_CATEGORY_VS_REST',training_program_export=True,training_export_scope='FROZEN_FAMILY_NOT_END_TO_END_SEARCH')
    def get_recipe(self,*,variant='selected'):
        if variant!='selected':return self._variant_view(variant).get_recipe()
        return dict(program=self.get_algorithm_program(),parameters=asdict(self.model_.params),preprocessing=copy.deepcopy(self.model_.encoder.state()),classes=list(self.classes_),role='FITTED_MODEL_RECIPE_NOT_AN_EXPORTED_TRAINING_PROCEDURE')
    def get_mutation_history(self):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        result=[];depth={}
        for e in self.report_.get('events',[]):
            if e['event']!='MUTATION':continue
            parent=e['parent_sha256'];depth.setdefault(parent,0);value=copy.deepcopy(e);value['parent_genealogical_depth']=depth[parent]
            if e['decision']=='ACCEPT':depth[e['child_sha256']]=depth[parent]+1
            value['retained_sha256']=e['child_sha256'] if e['decision']=='ACCEPT' else parent;value['retained_genealogical_depth']=depth[value['retained_sha256']];result.append(value)
        return result
    def report(self):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        from .scalable_variants import active_metadata
        result=copy.deepcopy(self.report_);result['variant_selection']=active_metadata(self)
        result['trace_scope']='Original fit history and costs; active variant source role is stated separately'
        return result
    def get_search_report(self):return self.report()
    def cost_report(self):
        report=self.report()
        native={}
        for event in report.get('events',()):
            if event['event']=='NATIVE_FIT_COMPLETED':
                phase=event['phase'];native[phase]=native.get(phase,0.)+event['seconds']
        return dict(scope='RECORDED_FIT_COST_NOT_NEW_MEASUREMENT',
                    native_fits=report.get('native_fits'),OOF_fits=report.get('OOF_fits'),
                    Optuna_trials=report.get('Optuna_trials'),
                    scalar_tree_reservations=report.get('scalar_trees'),
                    completed_scalar_trees=sum(e['actual_scalar_trees'] for e in report.get('events',()) if e['event']=='NATIVE_FIT_COMPLETED'),
                    total_fit_seconds=report.get('seconds'),native_fit_seconds_by_phase=native,
                    observed_resources=copy.deepcopy(report.get('resource_usage',{})),
                    native_time_scope='Native fit intervals only;OOF included. Excludes preprocessing,diagnosis bookkeeping,Optuna overhead and persistence;not a complete disjoint wall-time partition.',
                    new_training_from_this_query=0)
    def save_recipe(self,path,*,variant='selected'):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        if variant!='selected':return self._variant_view(variant).save_recipe(path)
        res=self._serialization_resources('Recipe snapshot')
        raw=json.dumps(self.get_recipe(),sort_keys=True,allow_nan=False)+'\n'
        res.reserve(len(raw.encode('utf8'))*4,'Serialized recipe')
        atomic_text(Path(path),raw)
        return hashlib.sha256(raw.encode('utf8')).hexdigest()
    def export_standalone(self,path,*,variant='selected',output_format='python'):
        """Export the fitted dependency-free predictor; this does not export training."""
        return self.export_python(path,variant=variant,output_format=output_format)
    def export_algorithm(self,path):
        """Export the frozen family trainer, distinct from fitted inference.

        train_encoded requires explicit TRAIN identity. This fragment does not
        rerun family screening, OOF, evolution or numerical selection.
        """
        from .training_fragment import export_algorithm
        return export_algorithm(self,path)
    @property
    def stop_reason_(self):
        if not hasattr(self,'model_'):raise AttributeError('stop_reason_ requires a fitted model')
        report=self.report_
        if 'stop_reason' in report:return report['stop_reason']
        if report.get('status')=='COMPLETED':return 'PLANNED_SEARCH_COMPLETE'
        if report.get('status')=='PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT':return 'PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT'
        return 'LEGACY_STOP_REASON_NOT_RECORDED'
    @property
    def feature_names_in_(self):
        if not hasattr(self,'model_'):raise AttributeError('feature_names_in_ requires a fitted model')
        names=self.model_.encoder.names
        return list(names) if names is not None else None
    def _score(self,X,y):
        self._ordinary_context_guard()
        if controlled_source(y):raise ValueError('Controlled sources cannot use ordinary scoring')
        if isinstance(y,(str,bytes,dict)) or getattr(y,'ndim',1)!=1 or not hasattr(y,'__len__'):
            raise ValueError('score requires nonempty aligned scalar targets')
        prediction=self.predict(X);n=len(prediction)
        if not n or len(y)!=n:raise ValueError('score requires nonempty aligned scalar targets')
        res=self._resources()
        if self.time_budget=='auto':res.configure_time(inference_seconds(n,1,0,0))
        regression=self.task=='regression'
        if regression:res.reserve(n*8,'Compact scoring targets');targets=array('d')
        correct=0;count=0
        for raw in y:
            if count%4096==0:res.check()
            if count>=n:raise ValueError('Score target iterator/length mismatch')
            value=scalar(raw) if regression else label_scalar(raw)
            if value is None:raise ValueError('Missing score targets are unsupported')
            if regression:
                if type(value) not in (int,float):raise ValueError('Regression score requires finite numeric targets')
                targets.append(value)
            else:
                predicted=prediction[count]
                correct+=value==(predicted.item() if hasattr(predicted,'item') else predicted)
            count+=1
        if count!=n:raise ValueError('Score target iterator/length mismatch')
        if not regression:return correct/n
        def residuals():
            for index,(target,predicted) in enumerate(zip(targets,prediction)):
                if index%4096==0:res.check()
                value=target-predicted.item()
                if not math.isfinite(value):raise ArithmeticError('Regression residual outside floating-point range')
                yield value
        error_scale=max(map(abs,residuals()),default=0.)
        if error_scale==0:return 1.
        if all(value==targets[0] for value in targets):return 0.
        mean=math.fsum(value/n for value in targets)
        def centered():
            for index,value in enumerate(targets):
                if index%4096==0:res.check()
                difference=value-mean
                if not math.isfinite(difference):raise ArithmeticError('Regression spread outside floating-point range')
                yield difference
        spread_scale=max(map(abs,centered()),default=0.)
        if spread_scale==0:return 0.
        error=error_scale*math.sqrt(math.fsum((value/error_scale)**2 for value in residuals())/n)
        spread=spread_scale*math.sqrt(math.fsum((value/spread_scale)**2 for value in centered())/n)
        ratio=error/spread;result=1-ratio*ratio
        if not math.isfinite(result):raise ArithmeticError('R2 outside floating-point range')
        return result
    def get_diagnostics(self):
        if not hasattr(self,'model_'):raise RuntimeError('Not fitted')
        return copy.deepcopy(getattr(self,'diagnostic_profiles_',{}))


class MorphTreeClassifier(_Scalable):
    task='classification'
    _estimator_type='classifier'
    def score(self,X,y):return self._score(X,y)
class MorphTreeRegressor(_Scalable):
    task='regression'
    _estimator_type='regressor'
    def score(self,X,y):return self._score(X,y)
    @property
    def predict_proba(self):raise AttributeError('Regression has no class probabilities')


def _standalone(X,packet,probabilities):
    import bisect,json,math,sys
    from numbers import Real
    if controlled_source(X):raise ValueError('Controlled sources cannot use ordinary standalone inference')
    model=packet['model'];schema=model['encoder'];classes=packet['classes'];task=model['task'];columns=schema['columns'];named=hasattr(X,'columns') and hasattr(X,'itertuples')
    names=list(X.columns) if named else None
    if schema['names'] is not None:
        if names is None or len(names)!=len(columns) or set(names)!=set(schema['names']):raise ValueError('Feature schema mismatch')
        positions={name:i for i,name in enumerate(names)};order=[positions[n] for n in schema['names']]
    else:
        if names is not None:raise ValueError('Positional TRAIN schema')
        order=list(range(len(columns)))
    if not len(X):raise ValueError('Nonempty inference input')
    allowance=memory_allowance('auto',memory_snapshot(available_memory())['effective_available_bytes'])
    if len(X)*(len(classes)*40+64 if probabilities else 40)>allowance:raise MemoryError('Standalone returned-result estimate exceeds currently available memory allowance')
    if probabilities and task=='regression':raise AttributeError('Regression has no probabilities')
    rows=X.itertuples(index=False,name=None) if named else iter(X)
    def category_key(v):
        if not isinstance(v,str):
            if type(v) is int:v=v if abs(v)>2**53 else float(v)
            else:
                v=float(v)
                if abs(v)>2**53 and v.is_integer():v=int(v)
            if v==0:v=0.0
        return json.dumps(['s' if isinstance(v,str) else 'n',v],ensure_ascii=False,allow_nan=False,separators=(',',':'))
    lookups={j:{category_key(json.loads(v)[1]):i+1 for i,v in enumerate(c['vocabulary'])} for j,c in enumerate(columns) if c['kind']=='categorical'}
    def sigmoid(v):
        if v>=0:return 1/(1+math.exp(-min(v,745.)))
        z=math.exp(max(v,-745.));return z/(1+z)
    out=[]
    for row in rows:
        if len(row)!=len(columns):raise ValueError('Feature count mismatch')
        values=[]
        for j,c in enumerate(columns):
            v=row[order[j]]
            if isinstance(v,str) and len(v)*32+512>allowance:raise MemoryError('Standalone scalar/key workspace exceeds available allowance')
            if type(v).__module__.split('.')[0]=='numpy' and hasattr(v,'item'):v=v.item()
            missing=v is None or (sys.modules.get('pandas') is not None and v is getattr(sys.modules['pandas'],'NA',None)) or isinstance(v,Real) and math.isnan(float(v))
            if not missing and not(isinstance(v,str) or isinstance(v,Real) and math.isfinite(float(v))):raise ValueError('Invalid scalar')
            if c['kind']=='numeric':
                if isinstance(v,str):raise ValueError('String in numeric feature')
                values.append(math.nan if missing else float(v))
            else:
                k=len(c['vocabulary']);key=None if missing else category_key(v)
                values.append(k+1 if missing else lookups[j].get(key,k+2))
        predictions=[]
        for member in model['members']:
            bins=[int(v) if j in lookups else 0 if math.isnan(v) else bisect.bisect_right(member['binner']['thresholds'][j],v)+1 for j,v in enumerate(values)]
            raw=list(member['base_score']) if task=='multiclass' else member['base_score']
            for index,(tree,step) in enumerate(zip(member['trees'],member['steps'])):
                nodes=tree['nodes'];node=nodes[0]
                for _ in range(len(nodes)+1):
                    if node['is_leaf']:break
                    j=node['feature'];b=bins[j];left=(b==node['threshold_bin']) if j in lookups else node['missing_left'] if b==0 else b<=node['threshold_bin'];node=nodes[node['left'] if left else node['right']]
                else:raise ValueError('Cyclic model')
                if task=='multiclass':
                    raw[index%len(classes)]+=step*node['value']
                    if index%len(classes)==len(classes)-1:
                        mean=math.fsum(raw)/len(raw);raw=[v-mean for v in raw]
                else:raw+=step*node['value']
            if task=='multiclass':
                weights=[math.exp(v-max(raw)) for v in raw];total=math.fsum(weights);predictions.append([v/total for v in weights])
            else:predictions.append(raw if task=='regression' else sigmoid(raw))
        if task=='multiclass':p=[math.fsum(v[j] for v in predictions)/len(predictions) for j in range(len(classes))]
        else:p=math.fsum(predictions)/len(predictions)
        if probabilities:out.append([1-p,p] if task=='binary' else p)
        elif task=='regression':out.append(p)
        else:out.append(classes[int(p>=.5)] if task=='binary' else classes[max(range(len(p)),key=p.__getitem__)])
    return out
