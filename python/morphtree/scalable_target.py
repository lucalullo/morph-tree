"""Compact target normalization without ndarray/Series.tolist or regression sort."""
from array import array
import sys
from .scalable_data import scalar


def label_scalar(value):
    if type(value).__module__.split('.')[0]=='numpy' and hasattr(value,'item'):value=value.item()
    # Class identifiers are not floating-point measurements. Preserve integer
    # identity above2**53 rather than merging distinct labels through float().
    return value if type(value) is int else scalar(value)


def detect_target(y,task,check,*,reserve=None):
    if getattr(y,'ndim',1)!=1 or isinstance(y,(str,bytes,dict)) or not hasattr(y,'__len__') or len(y)<4:raise ValueError('At least four one-dimensional scalar targets required')
    if task not in ('regression','classification','binary','multiclass'):raise ValueError('Explicit classifier/regressor intent required')
    # Width follows the addressable input,not a small arbitrary class ceiling.
    code='d' if task=='regression' else 'Q' if len(y)>2**32 else 'I'
    numeric=None;values=array(code);lookup={};count=0;vocabulary_bytes=0
    if reserve:reserve(len(y)*values.itemsize*(1 if task=='regression' else 2),'Target buffers before iteration')
    for value in y:
        if count%4096==0:check()
        if reserve and isinstance(value,str):reserve(len(y)*values.itemsize*2+vocabulary_bytes+len(value)*32+512,'Target scalar conversion workspace')
        value=scalar(value) if task=='regression' else label_scalar(value)
        if value is None:raise ValueError('Missing target is unsupported')
        is_numeric=type(value) in (int,float)
        if numeric is None:numeric=is_numeric
        if numeric!=is_numeric:raise ValueError('Target must be consistently string or numeric')
        if task=='regression' and not numeric:raise ValueError('Regression requires finite numeric targets')
        if task=='regression':values.append(value)
        else:
            if value not in lookup:
                vocabulary_bytes+=sys.getsizeof(value)+256
                if reserve:reserve(len(y)*values.itemsize*2+vocabulary_bytes,'Target vocabulary/codes/remap workspace')
                lookup[value]=len(lookup)
            values.append(lookup[value])
        count+=1
    if count!=len(y):raise ValueError('Target iterator/length mismatch')
    if task=='regression':return task,values,[]
    classes=sorted(lookup)
    if len(classes)<2 or task=='binary' and len(classes)!=2:raise ValueError('Distinct task-compatible labels required')
    task='binary' if len(classes)==2 else 'multiclass'
    remap=[None]*len(classes)
    for i,value in enumerate(classes):remap[lookup[value]]=i
    return task,array(code,(remap[v] for v in values)),classes


def compact_oof_train_indices(n, held, resources):
    """Validated complement in ascending source order,without Python-int sets.

    Index storage follows addressable rows; neither fold nor observations are
    sampled. Every returned index is unmarked and every held index is marked,
    which proves the logged disjointness without allocating a second large set.
    """
    if type(n) is not int or n < 2:raise ValueError('Positive OOF partition size')
    code='I' if n < 2**32 else 'Q'
    resources.reserve(n*(1+array(code).itemsize),'OOF membership mask and compact complement')
    marked=bytearray(n)
    for position,index in enumerate(held):
        if position%4096==0:resources.check()
        if type(index) is not int or not 0<=index<n or marked[index]:raise ValueError('Unique bounded held indices required')
        marked[index]=1
    result=array(code)
    for index in range(n):
        if index%4096==0:resources.check()
        if not marked[index]:result.append(index)
    if not result or len(result)==n:raise ValueError('OOF needs nonempty disjoint train and held partitions')
    return result
