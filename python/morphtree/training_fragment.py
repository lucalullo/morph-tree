"""Frozen native family training, never a substitute for evolutionary search.

The 2.0 public exporter provided an executor-bound train_encoded fragment.
This module supplies the scalable equivalent, with explicit TRAIN provenance,
finite resource guards, the same native booster and unchanged bootstrap/order.
Exporting never fits; calling the generated trainer is a new training operation.
"""
from array import array
from dataclasses import asdict,replace
import hashlib,json,time
from pathlib import Path


def _validate(spec):
    from .scalable import digest,verify_source_identity,_ENGINE_SHA256
    from .config import AlgorithmGenome,HyperParams,validate_task_genome
    verify_source_identity()
    fields={'schema','task','family','genome','parameters','encoder','classes','seed','runtime_manifest_sha256','scope'}
    if type(spec) is not dict or set(spec)!=fields or spec['schema']!='SCALABLE_FROZEN_TRAINING_1' or spec['scope']!='FROZEN_FAMILY_NOT_END_TO_END_SEARCH':raise ValueError('Exact frozen training fragment required')
    if spec['runtime_manifest_sha256']!=_ENGINE_SHA256:raise ValueError('Training fragment requires its exact source runtime')
    if spec['task'] not in ('binary','multiclass','regression') or spec['family'] not in ('boost','bag') or type(spec['seed']) is not int:raise ValueError('Known native task/family and resolved seed required')
    genome=AlgorithmGenome(**spec['genome']);params=HyperParams(**spec['parameters'])
    validate_task_genome(spec['task'],genome)
    return genome,params


def export_algorithm(estimator,path):
    from .scalable import digest,verify_source_identity,_ENGINE_SHA256
    from .persistence import atomic_text
    verify_source_identity()
    if not hasattr(estimator,'model_'):raise RuntimeError('Not fitted')
    res=estimator._serialization_resources('Frozen training program')
    seeds=[e['seed'] for e in estimator.report_.get('events',()) if e['event']=='NATIVE_FIT_STARTED']
    seed=estimator.report_.get('effective_seed',seeds[0] if seeds else estimator.random_state)
    spec=dict(schema='SCALABLE_FROZEN_TRAINING_1',task=estimator.task_,family=estimator.model_.family,
        genome=estimator.model_.genome.to_dict(),parameters=asdict(estimator.model_.params),
        encoder=estimator.model_.encoder.state(),classes=list(estimator.classes_),seed=seed,
        runtime_manifest_sha256=_ENGINE_SHA256,scope='FROZEN_FAMILY_NOT_END_TO_END_SEARCH')
    _validate(spec)
    text='''# MorphTree frozen native family training fragment; requires exact installed runtime.
# No family search, OOF, evolution, Optuna, holdout metric or inference weights.
from morphtree.training_fragment import train_encoded_fragment
SPEC = %r
SPEC_SHA256 = %r
EXPORT_SCOPE = 'FROZEN_FAMILY_NOT_END_TO_END_SEARCH'

def train_encoded(X, y, *, identity, memory_budget='auto', time_budget='auto', check=None):
    return train_encoded_fragment(SPEC, SPEC_SHA256, X, y, identity=identity,
        memory_budget=memory_budget, time_budget=time_budget, check=check)
''' % (spec,digest(spec))
    res.reserve(len(text.encode('utf8'))*4,'Frozen training source')
    atomic_text(Path(path),text);return hashlib.sha256(text.encode('utf8')).hexdigest()


def train_encoded_fragment(spec,expected_sha256,X,y,*,identity,memory_budget='auto',time_budget='auto',check=None):
    """Train exactly the frozen family on explicitly encoded TRAIN observations.

    X must be Encoder.transform(Table) from this runtime. identity is exactly
    {data: data_identity(X.source, compact_labels, 'TRAIN'),
     preprocessing_sha256: encoder_digest(exported_encoder)}.
    Targets are compact internal numeric labels, not raw class names. No rows
    are dropped; bag members use the original declared bootstrap procedure.
    """
    from .scalable import (digest,Family,MorphTreeClassifier,MorphTreeRegressor,
        managed_resources,_ENGINE_SHA256)
    from .scalable_data import Encoded,Encoder
    from .scalable_receipts import data_identity,encoder_digest,verify
    from .native_lab_backend import bind_backend
    from .booster import BinaryHistogramBooster
    from .prepared_bins import BinCache
    from ._experimental.sampling import bootstrap
    # Admission/context checks precede input traversal or a native component.
    if type(spec) is not dict or check is not None and not callable(check):raise ValueError('Typed fragment and optional resource callback required')
    obj=(MorphTreeRegressor if spec.get('task')=='regression' else MorphTreeClassifier)(
        random_state=spec.get('seed'),memory_budget=memory_budget,time_budget=time_budget,
        use_optuna=False,n_trials=0,max_mutations=0,refit_full=False,checkpoint_dir=None)
    obj._ordinary_context_guard()
    backend_factory=bind_backend()
    if digest(spec)!=expected_sha256:raise ValueError('Changed exported training program')
    genome,params=_validate(spec)
    if not isinstance(X,Encoded):raise ValueError('Explicit compact encoded TRAIN required')
    from .source_markers import controlled_source
    if controlled_source(X) or controlled_source(X.source) or controlled_source(y):raise ValueError('Controlled data require their original context')
    res=managed_resources(memory_budget,150 if time_budget=='auto' else time_budget,check)
    res.reserve(len(X)*(64+X.width*32),'Frozen native TRAIN workspace')
    target_codes=('d',) if spec['task']=='regression' else ('I','Q','q')
    if not isinstance(y,array) or y.typecode not in target_codes or not len(X) or len(X)!=len(y):raise ValueError('Aligned task-compatible native compact TRAIN labels required')
    enc=Encoder.from_state(spec['encoder'],res)
    wanted=dict(data=data_identity(X.source,y,'TRAIN'),preprocessing_sha256=encoder_digest(enc))
    if type(identity) is not dict or identity!=wanted:raise ValueError('Exact TRAIN/preprocessing identity required')
    transformed=enc.transform(X.source)
    if X.order!=transformed.order or X.maps!=transformed.maps or X._nominal_sizes!=transformed._nominal_sizes:raise ValueError('Encoded TRAIN differs from exported preprocessing')
    k=len(spec['classes']) if spec['task']!='regression' else 1
    if spec['task']=='regression':
        import math
        if spec['classes'] or any(not math.isfinite(v) for v in y):raise ValueError('Finite regression TRAIN targets')
    elif k<2 or spec['task']=='binary' and k!=2 or any(v!=int(v) or not 0<=v<k for v in y):raise ValueError('Task-compatible internal class indices required')
    # The same bound as normal training: output model + row workspace + histograms.
    from .resource_plan import fit_plan
    plan=fit_plan(len(X),X.width,k,params,0,0,False,res.available_bytes_snapshot,time_budget,res.bytes_limit)
    res.configure_time(plan['seconds']);res.reserve(plan['estimated_workspace_bytes'],'Frozen native family workspace')
    start=time.perf_counter();obj.last_attempt_=dict(status='IN_PROGRESS',native_fits=0,OOF_fits=0,scalar_trees=0,Optuna_trials=0,cache_hits=0,events=[],effective_seed=spec['seed'],training_scope=spec['scope'])
    from .scalable_receipts import Ledger
    obj._ledger=Ledger();obj._event('CALL_STARTED',engine_sha256=_ENGINE_SHA256,scope=spec['scope'],requested_task=spec['task'])
    obj._event('PROGRAM_FROZEN',genome_sha256=digest(genome.to_dict()))
    prepared=BinCache(min(32*1024**2,res.bytes_limit//8),res.check);models=[]
    try:
        for member in range(1 if spec['family']=='boost' else params.n_estimators):
            res.check();seed=spec['seed'];actual=params if spec['family']=='boost' else replace(params,n_estimators=1,learning_rate=1.)
            if spec['family']=='bag':
                ids=bootstrap(y,spec['task'],seed+member+1);table=X.source.take(ids);yy=array(y.typecode,(y[i] for i in ids));xx=enc.transform(table)
            else:table=X.source;xx=X;yy=y
            data=data_identity(X.source,y,'TRAIN');preprocessing=encoder_digest(enc)
            key=digest(dict(data=data,preprocessing_sha256=preprocessing,task=spec['task'],family=spec['family'],genome=genome.to_dict(),params=asdict(params),seed=seed,engine_sha256=_ENGINE_SHA256))
            rows=hashlib.sha256(memoryview(table.indices)).hexdigest() if table.indices is not None else digest(dict(range_start=0,range_stop=len(table)))
            labels=hashlib.sha256(memoryview(yy)).hexdigest();component=digest(dict(request=key,member=member,effective_rows=rows,effective_labels=labels,seed=seed+member,params=asdict(actual)))
            count=actual.n_estimators*(k if spec['task']=='multiclass' else 1)
            obj.last_attempt_['native_fits']+=1;obj.last_attempt_['scalar_trees']+=count
            obj._event('NATIVE_FIT_STARTED',ordinal=member+1,phase='FROZEN_TRAIN',rows=len(yy),source_rows=len(X),family=spec['family'],cache_key=key,component_key=component,effective_rows_sha256=rows,effective_labels_sha256=labels,seed=seed+member,params_sha256=digest(asdict(actual)),preprocessing_sha256=preprocessing,program_sha256=digest(genome.to_dict()),role='TRAIN',reserved_scalar_trees=count)
            model=BinaryHistogramBooster(genome,actual,seed+member,backend_factory(xx._nominal_sizes),memory_enabled=False,objective='softmax' if spec['task']=='multiclass' else 'squared_error' if spec['task']=='regression' else 'binary_logloss',n_classes=k if spec['task']=='multiclass' else None)
            token=prepared.bind(xx,dict(source_sha256=data['source_sha256'],schema_sha256=data['schema_sha256'],effective_rows_sha256=rows,effective_labels_sha256=labels,preprocessing_sha256=preprocessing,engine_sha256=_ENGINE_SHA256),'TRAIN')
            tick=time.perf_counter();model.fit(xx,yy,callback=lambda *_:res.check() or True,prepared_data=token)
            if len(model.trees)!=count:raise ValueError('Incomplete frozen native component')
            obj._event('NATIVE_FIT_COMPLETED',ordinal=member+1,phase='FROZEN_TRAIN',seconds=time.perf_counter()-tick,component_key=component,actual_scalar_trees=len(model.trees));models.append(model)
        obj.model_=Family(spec['task'],genome,params,enc,models,spec['family']);obj.classes_=list(spec['classes']);obj.task_=spec['task'];obj.n_features_in_=X.width;obj.best_family_=spec['family'];obj.best_params_=asdict(params)
        obj._event('FINAL_MODEL',input_rows=len(X),selection_TRAIN_rows=len(X),SEARCH_rows=0,final_fit_source_rows=len(X),refit_full=False,SEARCH_primary_before_refit=None,selection_metric_is_independent=False,nominal_columns=len(X._nominal_sizes),source_storage_bytes=X.source.storage_bytes,all_rows_accounted=True)
        obj.last_attempt_.update(status='COMPLETED',seconds=time.perf_counter()-start,stop_reason='FROZEN_PROGRAM_TRAINING_COMPLETE')
        obj.last_attempt_['receipt_verification']=verify(obj.last_attempt_['events']);obj.report_=obj.last_attempt_
        return obj
    finally:prepared.clear()
