"""Research-only numeric-space admission, reusing actual native activity rules.

Potentially active is not guaranteed sensitive on every dataset. Frozen topology,
binning and learning-budget dimensions are not final numeric opportunities here.
No historical tuning space or production API is modified.
"""
from dataclasses import asdict,fields
from morphtree.config import AlgorithmGenome,HyperParams

FROZEN={'max_depth','max_leaves','max_bins','n_estimators'}

def schema(family,task,genome):
    if family not in ('boost','bag') or task not in ('binary','multiclass','regression') or not isinstance(genome,AlgorithmGenome):raise ValueError('Explicit native family/task/genome')
    if task!='binary' and genome!=AlgorithmGenome():raise ValueError('Current native nonbinary API supports fixed Base genome only')
    active=set(genome.active_parameters());known={f.name for f in fields(HyperParams)};frozen=set(FROZEN)
    if family=='bag':
        frozen.add('learning_rate')
        # Reset members have one stage: decay multiplies iteration0, hence zero.
        active.discard('update_decay')
    return {name:{'status':'INACTIVE' if name not in active else 'FROZEN_FORM_OR_BUDGET' if name in frozen else 'POTENTIALLY_ACTIVE','condition':'AlgorithmGenome.active_parameters + fixed-form/budget boundary','data_dependent':True} for name in sorted(known)}

def admit_space(space,*,family,task,genome,drop_inactive=False):
    if not isinstance(space,dict) or not space or type(drop_inactive) is not bool:raise ValueError('Nonempty explicit future space')
    rules=schema(family,task,genome);accepted={};excluded={}
    for name,spec in space.items():
        if name not in rules:raise ValueError('Unknown parameter: '+str(name))
        status=rules[name]['status']
        if status=='FROZEN_FORM_OR_BUDGET':raise ValueError('Structural/budget dimension is frozen: '+name)
        if status=='INACTIVE':
            if not drop_inactive:raise ValueError('Inactive under actual genome: '+name)
            excluded[name]='INACTIVE_UNDER_FROZEN_GENOME';continue
        if type(spec) not in (list,tuple) or len(spec)!=2:raise ValueError('Bound pair required')
        lo,hi=spec
        if type(lo) not in (int,float) or type(hi) not in (int,float) or not lo<hi:raise ValueError('Increasing finite native parameter bounds')
        for v in (lo,hi):HyperParams(**(asdict(HyperParams())|{name:v}))
        accepted[name]=list(spec)
    if not accepted:raise ValueError('No active numeric opportunity remains')
    return {'space':accepted,'excluded':excluded,'mode':'RESEARCH_ONLY_NOT_INSTALLED_OR_HISTORICAL_SPACE_CHANGE'}
