"""Bounded, call-local reuse of exact TRAIN preprocessing, never learner fits.

The facade supplies identities from its owned copied buffers. Tokens are bound
to one encoded input object. Source, ordered sample, labels, schema, encoder,
role, engine, bin policy and bin count all participate in the reuse key.
Entries are private and not persisted. The cap covers retained bins/thresholds
and conservative entry overhead, not all fit/process allocations.
"""
from array import array
from collections import OrderedDict
import hashlib
import sys
import time
from .data import Binner


IDENTITIES=('source_sha256','schema_sha256','effective_rows_sha256',
            'effective_labels_sha256','preprocessing_sha256','engine_sha256')


def fingerprint(binner, bins):
    h=hashlib.sha256()
    h.update(binner.policy.encode('ascii'));h.update(binner.max_bins.to_bytes(8,'little'))
    for cuts in binner.thresholds:
        h.update(len(cuts).to_bytes(8,'little'));h.update(memoryview(array('d',cuts)))
    for column in bins:
        h.update(column.typecode.encode('ascii'));h.update(bytes((bool(getattr(column,'nominal',False)),)))
        h.update(len(column).to_bytes(8,'little'));h.update(memoryview(column))
    return h.hexdigest()


def retained_bytes(binner,bins,key):
    # Counting shared small integers more than once is deliberately conservative.
    def size(v):return sys.getsizeof(v)+(sum(size(x) for x in v) if type(v) is tuple else 0)
    return (2048+size(key)
            +sys.getsizeof(binner)+sys.getsizeof(binner.__dict__)
            +sys.getsizeof(binner.thresholds)
            +sum(sys.getsizeof(cuts)+sum(sys.getsizeof(v) for v in cuts) for cuts in binner.thresholds)
            +sys.getsizeof(bins)+sum(sys.getsizeof(c)+sys.getsizeof(c.__dict__) for c in bins))


class BinCache:
    def __init__(self,limit_bytes,check):
        if type(limit_bytes) is not int or limit_bytes<0 or not callable(check):raise ValueError('Finite cache allowance/check')
        self.limit_bytes=limit_bytes;self.check=check;self.entries=OrderedDict();self.bytes=0
        self.counts=dict(hits=0,misses=0,bypassed=0,evictions=0,peak_retained_bytes=0,reused_rows=0,preparation_seconds=0.,integrity_seconds=0.)
    def bind(self,X,identity,role):
        if type(identity) is not dict or set(identity)!=set(IDENTITIES):raise ValueError('Exact owned-source identity')
        if any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in identity.values()):raise ValueError('SHA256 identity fields')
        if role not in ('TRAIN','OOF_TRAIN','TRAIN_FULL'):raise ValueError('Learning role required; no SEARCH/EXTERNAL preparation')
        if not getattr(X,'_compact_columns',False):raise ValueError('Owned compact encoded input required')
        return _Bound(self,X,tuple(identity[n] for n in IDENTITIES)+(role,))
    def _get(self,X,base,genome,params,backend):
        self.check()
        nominal=tuple(sorted(X._nominal_sizes.items()))
        if nominal!=tuple(sorted(backend.nominal_sizes.items())):raise ValueError('Prepared bins/backend nominal schema mismatch')
        key=base+(len(X),X.width,genome.binning_policy,params.max_bins,nominal)
        if key in self.entries:
            binner,bins,size,stamp=self.entries[key];tick=time.perf_counter()
            if fingerprint(binner,bins)!=stamp:raise ValueError('Changed call-local prepared buffers')
            self.counts['integrity_seconds']+=time.perf_counter()-tick
            self.entries.move_to_end(key);self.counts['hits']+=1;self.counts['reused_rows']+=len(X)
            self.check();return binner,bins
        self.counts['misses']+=1;tick=time.perf_counter()
        binner=Binner(params.max_bins,genome.binning_policy).fit(X);bins=backend.bin(binner,X)
        self.counts['preparation_seconds']+=time.perf_counter()-tick;self.check()
        size=retained_bytes(binner,bins,key)
        if size>self.limit_bytes:
            self.counts['bypassed']+=1;return binner,bins
        while self.bytes+size>self.limit_bytes:
            _,(_,_,removed,_)=self.entries.popitem(last=False);self.bytes-=removed;self.counts['evictions']+=1
        self.entries[key]=(binner,bins,size,fingerprint(binner,bins));self.bytes+=size
        self.counts['peak_retained_bytes']=max(self.bytes,self.counts['peak_retained_bytes']);self.check();return binner,bins
    def report(self):return dict(limit_bytes=self.limit_bytes,retained_bytes=self.bytes,entries=len(self.entries),**self.counts)
    def clear(self):self.entries.clear();self.bytes=0


class _Bound:
    def __init__(self,owner,X,base):self.owner,self.X,self.base=owner,X,base
    def get(self,X,genome,params,backend):
        if X is not self.X:raise ValueError('Prepared-data token belongs to a different input object')
        return self.owner._get(X,self.base,genome,params,backend)
