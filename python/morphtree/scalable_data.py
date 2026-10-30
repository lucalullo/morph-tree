"""Private scalable representation. No whole DataFrame/ndarray list conversion.

Source dictionaries encode input storage only. Learner vocabularies are fitted
again on each TRAIN/fold; SEARCH-only categories become unknown at validation.
"""
from array import array
from bisect import bisect_right
from collections.abc import Sequence
import json
import math
import os
from numbers import Real
import sys
import time
from .source_markers import controlled_source


def scalar(v,*,preserve_integral=False):
    kind=type(v)
    if kind is str or v is None:return v
    if kind is int and preserve_integral:return v
    if kind is float or kind is int or kind is bool:
        f=float(v)
        if math.isnan(f):return None
        if math.isfinite(f):return f
        raise ValueError('Finite numeric/string category/None/NaN/nullable scalar required')
    if type(v).__module__.split('.')[0]=='numpy' and hasattr(v,'item'):v=v.item()
    if v is None or (sys.modules.get('pandas') is not None and v is getattr(sys.modules['pandas'],'NA',None)):return None
    if preserve_integral and type(v) is int:return v
    if isinstance(v,Real):
        f=float(v)
        if math.isnan(f):return None
        if math.isfinite(f):return f
    if isinstance(v,str):return v
    raise ValueError('Finite numeric/string category/None/NaN/nullable scalar required')


def category_key(v):
    if not isinstance(v,str):
        if type(v) is int:v=v if abs(v)>2**53 else float(v)
        else:
            v=float(v)
            if abs(v)>2**53 and v.is_integer():v=int(v)
        if v==0:v=0.0
    return json.dumps(['s' if isinstance(v,str) else 'n',v],ensure_ascii=False,allow_nan=False,separators=(',',':'))


def canonical_category_key(key):
    """Exact numeric identity, including compatible legacy float key spelling."""
    entry=json.loads(key)
    if type(entry) is not list or len(entry)!=2:raise ValueError('Malformed typed category')
    tag,value=entry
    if not (tag=='s' and type(value) is str or tag=='n' and type(value) in (int,float)):raise ValueError('Malformed typed category')
    value=scalar(value,preserve_integral=True)
    if value is None:raise ValueError('Missing value is not a vocabulary entry')
    canonical=category_key(value)
    try:legacy=json.dumps([tag,value if tag=='s' else float(value)],ensure_ascii=False,allow_nan=False,separators=(',',':'))
    except OverflowError:legacy=None
    if key not in (canonical,legacy):raise ValueError('Noncanonical typed category')
    return canonical


def resident_bytes():
    """Read-only OS accounting; no learner arithmetic outside Python."""
    try:
        if os.name=='nt':
            import ctypes
            class Counters(ctypes.Structure):_fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
            c=Counters();c.cb=ctypes.sizeof(c)
            if not ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(c),c.cb):return None
            return c.rss
        with open('/proc/self/statm',encoding='ascii') as handle:
            resident_pages=int(handle.read().split()[1])
        return resident_pages*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,AttributeError):return None


class Input:
    """Inspect metadata first, then access one column or bounded block."""
    def __init__(self,X):
        if controlled_source(X):raise ValueError('Controlled sources cannot use ordinary input or inference')
        self.X=X;self.frame=hasattr(X,'columns') and hasattr(X,'iloc');self.fitted_kinds=None
        if hasattr(X,'format') and hasattr(X,'tocsr'):raise ValueError('Sparse input requires a separate qualified path')
        if hasattr(X,'shape'):
            if len(X.shape)!=2:raise ValueError('X must have exactly two dimensions')
            self.n,self.width=map(int,X.shape)
        elif isinstance(X,(list,tuple)):
            self.n=len(X);self.width=len(X[0]) if self.n and isinstance(X[0],(list,tuple,array)) else 0
            if any(not isinstance(r,(list,tuple,array)) or len(r)!=self.width for r in X):raise ValueError('Rectangular rows required')
        else:raise ValueError('Supported dense table, ndarray or DataFrame required')
        if self.n<1 or self.width<1:raise ValueError('X must contain rows and columns')
        self.names=tuple(X.columns) if self.frame else None
        if self.names is not None and (len(set(self.names))!=self.width or any(not isinstance(n,str) or not n for n in self.names)):raise ValueError('Unique nonempty string names required')
    def column(self,j,start=0,stop=None):
        stop=self.n if stop is None else stop
        if self.frame:return iter(self.X.iloc[start:stop,j].array)
        if hasattr(self.X,'shape'):return iter(self.X[start:stop,j])
        return (self.X[i][j] for i in range(start,stop))
    def forced_categorical(self,j):return self.fitted_kinds[j]=='categorical' if self.fitted_kinds is not None else self.frame and str(self.X.dtypes.iloc[j])=='category'


class Resources:
    def __init__(self,bytes_limit,seconds,check=None):
        if type(bytes_limit) is not int or bytes_limit<1 or not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite resource allowance')
        self.bytes_limit=bytes_limit;self.started=time.monotonic();self.deadline=self.started+seconds;self.external=check;self.base_rss=resident_bytes();self.peak_rss=self.base_rss;self.last_memory_check=0.;self.pressure_reader=None;self.base_commit=None;self.peak_commit=None
    def configure_time(self,seconds):
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite resource allowance')
        self.deadline=self.started+seconds
    def check(self):
        if time.monotonic()>self.deadline:raise TimeoutError('Finite scalable operation deadline')
        if time.monotonic()-self.last_memory_check>.05:
            self.last_memory_check=time.monotonic();rss=resident_bytes()
            if rss is not None:
                self.peak_rss=max(self.peak_rss or 0,rss)
                if self.base_rss is not None and rss-self.base_rss>self.bytes_limit:raise MemoryError('Observed process RSS increase exceeds operation allowance; no rows were sampled')
            if self.pressure_reader:
                observation=self.pressure_reader();commit=observation.get('process_private_bytes')
                if commit is not None:
                    self.peak_commit=max(self.peak_commit or 0,commit)
                    if self.base_commit is not None and commit-self.base_commit>self.bytes_limit:raise MemoryError('Observed process committed-memory increase exceeds operation allowance; no rows were sampled')
                if observation['effective_available_bytes']<=0:raise MemoryError('Active RAM/commit/address-space headroom exhausted; no rows were sampled')
        if self.external:self.external()
    def reserve(self,n,description):
        if n>self.bytes_limit:raise MemoryError(f'{description} requires estimated {n} bytes, allowance {self.bytes_limit}; no rows were sampled')
        self.check()


class Rows(Sequence):
    _compact_columns=True
    def __len__(self):return self.n
    def __getitem__(self,i):
        if isinstance(i,slice):return [self[j] for j in range(*i.indices(self.n))]
        if i<0:i+=self.n
        if not 0<=i<self.n:raise IndexError(i)
        return Row(self,i)
    def column(self,j):return (self.value(i,j) for i in range(self.n))


class Row(Sequence):
    def __init__(self,table,i):self.table=table;self.i=i
    def __len__(self):return self.table.width
    def __getitem__(self,j):
        if isinstance(j,slice):return [self[k] for k in range(*j.indices(len(self)))]
        if j<0:j+=len(self)
        if not 0<=j<len(self):raise IndexError(j)
        return self.table.value(self.i,j)


class Table(Rows):
    def __init__(self,columns,kinds,vocab,n,names,indices=None):
        self.columns,self.kinds,self.vocab,self.n,self.names,self.indices=columns,kinds,vocab,n,names,indices;self.width=len(columns)
    @classmethod
    def read(cls,source,resources,start=0,stop=None):
        source=source if isinstance(source,Input) else Input(source);stop=source.n if stop is None else stop;n=stop-start
        if not 0<=start<stop<=source.n:raise ValueError('Nonempty bounded block')
        name_bytes=sum(sys.getsizeof(name)+8*len(name) for name in source.names or ())
        base_bytes=n*source.width*10+source.width*256+name_bytes
        resources.reserve(base_bytes,'Compact source columns and schema names')
        columns=[];kinds=[];vocab=[];vocab_bytes=0
        for j in range(source.width):
            resources.check();first=None
            for i,value in enumerate(source.column(j,start,stop)):
                if i%4096==0:resources.check()
                if isinstance(value,str):resources.reserve(base_bytes+vocab_bytes+len(value)*32+512,'Initial scalar/JSON key workspace')
                first=scalar(value)
                if first is not None:break
            forced=source.forced_categorical(j);cat=forced or isinstance(first,str);a=array('I' if cat else 'd');lookup={};words=[]
            for i,v in enumerate(source.column(j,start,stop)):
                if i%4096==0:resources.check()
                if isinstance(v,str):resources.reserve(base_bytes+vocab_bytes+len(v)*32+512,'Scalar/JSON key workspace before allocation')
                v=scalar(v,preserve_integral=cat)
                if cat:
                    if v is None:a.append(0);continue
                    if not forced and not isinstance(v,str):raise ValueError('Mixed string/numeric column')
                    key=category_key(v)
                    if key not in lookup:
                        vocab_bytes+=sys.getsizeof(key)+96;resources.reserve(base_bytes+vocab_bytes,'Source columns and category dictionaries')
                        lookup[key]=len(words)+1;words.append(key)
                    a.append(lookup[key])
                else:
                    if isinstance(v,str):raise ValueError('Mixed string/numeric column')
                    a.append(math.nan if v is None else v)
            columns.append(a);kinds.append('categorical' if cat else 'numeric');vocab.append(tuple(words))
        return cls(tuple(columns),tuple(kinds),tuple(vocab),n,source.names)
    def take(self,indices):
        ids=array('I' if len(self.columns[0])<2**32 else 'Q',(self.indices[i] if self.indices is not None else i for i in indices))
        return Table(self.columns,self.kinds,self.vocab,len(ids),self.names,ids)
    def value(self,i,j):return self.columns[j][self.indices[i] if self.indices is not None else i]
    def column(self,j):
        c=self.columns[j]
        return iter(c) if self.indices is None else (c[i] for i in self.indices)
    @property
    def storage_bytes(self):return sum(sys.getsizeof(c) for c in self.columns)+sum(sys.getsizeof(s) for v in self.vocab for s in v)


class Encoder:
    """One native nominal feature per category column, TRAIN-only vocabulary."""
    def fit(self,table):
        self.names=table.names;self.columns=[]
        for j,kind in enumerate(table.kinds):
            if kind=='numeric':self.columns.append(dict(kind='numeric'));continue
            observed=set(table.column(j));vocab=sorted({canonical_category_key(table.vocab[j][i-1]) for i in observed if i})
            self.columns.append(dict(kind='categorical',vocabulary=vocab))
        return self
    def state(self):return dict(schema='SCALABLE_TRAIN_ENCODER_1',names=list(self.names) if self.names is not None else None,columns=self.columns)
    @classmethod
    def from_state(cls,state,resources):
        if type(state) is not dict or set(state)!={'schema','names','columns'} or state['schema']!='SCALABLE_TRAIN_ENCODER_1':raise ValueError('Exact private encoder schema')
        cols=state['columns'];names=state['names']
        if type(cols) is not list or not cols:raise ValueError('Nonempty column schema')
        if names is not None and (type(names) is not list or len(names)!=len(cols) or len(set(names))!=len(names) or any(type(n) is not str or not n for n in names)):raise ValueError('Invalid encoder names')
        schema_bytes=len(cols)*256+sum(sys.getsizeof(name)+8*len(name) for name in names or ())
        resources.reserve(schema_bytes,'Decoded encoder metadata before copies')
        for c in cols:
            if c==dict(kind='numeric'):continue
            if type(c) is not dict or set(c)!={'kind','vocabulary'} or c['kind']!='categorical':raise ValueError('Invalid category schema')
            v=c['vocabulary']
            if type(v) is not list or any(type(s) is not str for s in v) or any(v[i]>=v[i+1] for i in range(len(v)-1)):raise ValueError('Canonical typed vocabulary required')
            schema_bytes+=len(v)*128;canonical=set()
            for key in v:
                schema_bytes+=sys.getsizeof(key)*3
                resources.reserve(schema_bytes+len(key)*32+512,'Typed vocabulary validation before JSON allocation')
                normalized=canonical_category_key(key)
                if normalized in canonical:raise ValueError('Duplicate equivalent numeric categories')
                canonical.add(normalized)
        obj=cls();obj.names=tuple(names) if names is not None else None;obj.columns=cols;return obj
    def transform(self,table):
        if self.names is None:
            if table.names is not None:raise ValueError('Positional TRAIN requires positional prediction')
            order=list(range(table.width))
        else:
            if table.names is None or set(table.names)!=set(self.names):raise ValueError('Feature schema mismatch')
            positions={name:i for i,name in enumerate(table.names)};order=[positions[n] for n in self.names]
        if len(order)!=len(self.columns):raise ValueError('Column count mismatch')
        maps={};nominal={}
        for j,(source,c) in enumerate(zip(order,self.columns)):
            if c['kind']!=table.kinds[source]:
                # An all-missing prediction column has no inferred string type.
                if not(c['kind']=='categorical' and all(math.isnan(v) for v in table.column(source))):raise ValueError('Feature type mismatch')
            if c['kind']=='categorical':
                lookup={canonical_category_key(key):i+1 for i,key in enumerate(c['vocabulary'])};k=len(lookup);nominal[j]=k
                maps[j]=[k+1]+[lookup.get(canonical_category_key(key),k+2) for key in table.vocab[source]]
        return Encoded(table,order,maps,nominal)


class Encoded(Rows):
    def __init__(self,table,order,maps,nominal):self.source=table;self.order=order;self.maps=maps;self._nominal_sizes=nominal;self.n=table.n;self.width=len(order)
    def value(self,i,j):
        v=self.source.value(i,self.order[j])
        return (self.maps[j][0] if self.source.kinds[self.order[j]]=='numeric' else self.maps[j][int(v)]) if j in self.maps else v
    def column(self,j):
        values=self.source.column(self.order[j])
        if j not in self.maps:return values
        if self.source.kinds[self.order[j]]=='numeric':return (self.maps[j][0] for _ in values)
        return (self.maps[j][int(v)] for v in values)
