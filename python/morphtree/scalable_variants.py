"""Retained genuine model variants. Never train or fabricate absent weights.

pure and tuned are verified TRAIN-stage models. selected is the default final
model, normally refitted on all eligible input. A bundle stores each distinct
weight packet once and the original fit history once; no dataset is embedded.
"""
import copy,hashlib,json
from dataclasses import asdict,replace
from pathlib import Path
from .persistence import atomic_text

SCHEMA='SCALABLE_PRIVATE_VARIANTS_1'
NAMES=frozenset(('selected','pure','tuned'))
BASE_KEYS=frozenset(('schema','model','classes','report','receipt_sha256','evidence','constructor','constructor_sha256'))


def _role(report):
    if report.get('status')=='PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT':return 'TRAIN'
    final=next((e for e in reversed(report.get('events',())) if e['event']=='FINAL_MODEL'),None)
    return 'TRAIN_FULL' if final and final['refit_full'] else 'TRAIN' if final else 'UNKNOWN'


def catalog(estimator):
    if not hasattr(estimator,'model_'):raise RuntimeError('Not fitted')
    if hasattr(estimator,'_retained_variants'):return estimator._retained_variants
    result={'selected':dict(model=estimator.model_,origin='SELECTED_FITTED_MODEL',source_role=_role(estimator.report_))}
    pure=getattr(estimator,'_research_structural_before_numeric',None)
    if pure is not None:
        result['pure']=dict(model=pure,origin='PRE_FINAL_NUMERIC_SEARCH',source_role='TRAIN')
        tuned=getattr(estimator,'_research_selected_before_refit',None)
        if tuned is not None and tuned is not pure:
            result['tuned']=dict(model=tuned,origin='POST_FINAL_NUMERIC_SEARCH',source_role='TRAIN')
    return result


def resolve(estimator,name):
    if type(name) is not str or name not in NAMES:raise ValueError('Variant must be selected, pure or tuned')
    if name=='selected':name=getattr(estimator,'_variant_selection','selected')
    variants=catalog(estimator)
    if name not in variants:raise ValueError('Variant has no retained genuine weights; no implicit training is performed')
    return name,variants[name]


def active_metadata(estimator):
    name,item=resolve(estimator,'selected')
    return dict(name=name,origin=item['origin'],source_role=item['source_role'],additional_training=False)


def describe(estimator):
    from .scalable import digest
    variants=catalog(estimator);estimator._serialization_resources('Variant inspection',families=[v['model'] for v in variants.values()])
    selected=resolve(estimator,'selected')[0]
    return {name:dict(origin=v['origin'],source_role=v['source_role'],params=asdict(v['model'].params),model_sha256=digest(v['model'].packet()),exportable=True,is_selected=name==selected,additional_training=False) for name,v in variants.items()}


def view(estimator,name):
    name,item=resolve(estimator,name);other=copy.copy(estimator)
    other._retained_variants=catalog(estimator);other._variant_selection=name
    other.model_=item['model'];other.best_family_=other.model_.family;other.best_params_=asdict(other.model_.params)
    return other


def _binding(selected,variants,models):
    return dict(selected_variant=selected,variants=variants,models=models)


def save_bundle(estimator,path,variant):
    from .scalable import digest,verify_receipts
    if type(variant) is not str or variant not in NAMES|{'all'}:raise ValueError('Variant must be all, selected, pure or tuned')
    available=catalog(estimator)
    if variant=='all':names=list(available);selected=resolve(estimator,'selected')[0]
    else:selected=resolve(estimator,variant)[0];names=[selected]
    # Preserve existing single-model formats for old objects without retained
    # phase weights; do not relabel their final model as an unrecorded pure fit.
    if names==['selected'] and set(available)=={'selected'}:return None
    res=estimator._serialization_resources('Variant bundle',families=[available[n]['model'] for n in names])
    verify_receipts(estimator.report_['events'])
    models={};variants={}
    for name in names:
        res.check();item=available[name];packet=item['model'].packet();identity=digest(packet)
        models.setdefault(identity,packet)
        variants[name]=dict(origin=item['origin'],source_role=item['source_role'],model_sha256=identity)
    primary=models.pop(variants[selected]['model_sha256'])
    constructor=estimator._constructor_state()
    packet=dict(schema=SCHEMA,model=primary,classes=estimator.classes_,report=estimator.report_,receipt_sha256=digest(estimator.report_),
                evidence=dict(diagnostic_profiles=getattr(estimator,'diagnostic_profiles_',{})),constructor=constructor,constructor_sha256=digest(constructor),
                selected_variant=selected,variants=variants,models=models,variants_sha256=digest(_binding(selected,variants,models)))
    raw=json.dumps(packet,sort_keys=True,allow_nan=False)+'\n';res.reserve(len(raw.encode('utf8'))*4,'Encoded unique-weight variant bundle')
    atomic_text(Path(path),raw);return hashlib.sha256(raw.encode('utf8')).hexdigest()


def restore_bundle(cls,packet,res,limit,file_sha256,expected_checksum_supplied,memory_budget):
    from .scalable import digest
    required=BASE_KEYS|{'selected_variant','variants','models','variants_sha256'}
    if set(packet)!=required or packet['schema']!=SCHEMA:raise ValueError('Exact variant bundle envelope required')
    variants=packet['variants'];models=packet['models'];selected=packet['selected_variant']
    if type(variants) is not dict or not variants or not set(variants)<=NAMES or type(models) is not dict or len(models)>2 or type(selected) is not str or selected not in variants:
        raise ValueError('Finite named retained variants required')
    if digest(_binding(selected,variants,models))!=packet['variants_sha256']:raise ValueError('Changed variant bundle identity')
    for name,item in variants.items():
        if type(item) is not dict or set(item)!={'origin','source_role','model_sha256'}:raise ValueError('Exact variant identity/role required')
        expected='SELECTED_FITTED_MODEL' if name=='selected' else 'PRE_FINAL_NUMERIC_SEARCH' if name=='pure' else 'POST_FINAL_NUMERIC_SEARCH'
        if item['origin']!=expected or item['source_role'] not in (('TRAIN','TRAIN_FULL','UNKNOWN') if name=='selected' else ('TRAIN',)):
            raise ValueError('Variant origin/source role mismatch')
        h=item['model_sha256']
        if type(h) is not str or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Variant model digest required')
    primary=variants[selected]['model_sha256']
    if digest(packet['model'])!=primary:raise ValueError('Changed selected variant weights')
    if set(models)!={v['model_sha256'] for v in variants.values()}-{primary}:raise ValueError('Missing/unreferenced retained weight packet')
    if any(type(v) is not dict or digest(v)!=h for h,v in models.items()):raise ValueError('Changed retained variant weights')
    base={k:packet[k] for k in BASE_KEYS};base['schema']='SCALABLE_PRIVATE_MODEL_3'
    obj=cls._restore_packet(base,res,limit,file_sha256,expected_checksum_supplied,memory_budget)
    if 'selected' in variants and variants['selected']['source_role']!=_role(packet['report']):raise ValueError('Selected role must match original fit finalization')
    families={primary:obj.model_}
    for identity,model in models.items():
        res.check();part=cls._restore_packet(dict(base,model=model),res,limit,file_sha256,expected_checksum_supplied,memory_budget)
        if (part.task_,part.classes_,part.model_.family,part.model_.genome)!=(obj.task_,obj.classes_,obj.model_.family,obj.model_.genome):raise ValueError('Variant task/family/frozen program mismatch')
        if part.model_.encoder.names!=obj.model_.encoder.names or [c['kind'] for c in part.model_.encoder.columns]!=[c['kind'] for c in obj.model_.encoder.columns]:raise ValueError('Variant feature schema mismatch')
        families[identity]=part.model_
    if 'tuned' in variants:
        identity=variants['tuned']['model_sha256'];family=families[identity]
        if identity!=packet['report'].get('last_verified_anchor',{}).get('model_sha256'):raise ValueError('Tuned weights must match retained TRAIN anchor')
        if not any(e['event']=='OPTUNA_COMPLETE' and e['parameters']==asdict(family.params) for e in packet['report']['events']):raise ValueError('Tuned variant needs a completed numerical trial')
    if 'pure' in variants:
        identity=variants['pure']['model_sha256'];family=families[identity]
        capacity=family.params if family.family=='boost' else replace(family.params,n_estimators=1,learning_rate=1.)
        if not any(e['event']=='NATIVE_FIT_STARTED' and e['phase']=='SCREEN' and e['family']==family.family and e['params_sha256']==digest(asdict(capacity)) for e in packet['report']['events']):raise ValueError('Pure variant needs qualified untuned capacity')
        history=[e for e in packet['report']['events'] if e['event']=='MUTATION']
        if history:
            last=history[-1];expected=last['child_sha256'] if last['decision']=='ACCEPT' else last['parent_sha256']
            if identity!=expected:raise ValueError('Pure variant must match retained structural lineage')
    obj._retained_variants={name:dict(model=families[v['model_sha256']],origin=v['origin'],source_role=v['source_role']) for name,v in variants.items()}
    obj._variant_selection=selected;return obj
