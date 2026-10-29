"""Owned compact buffers and a bounded content-keyed tree-layout cache."""
from array import array
from collections import OrderedDict
import ctypes,struct,sys,threading
from ..scalable_backend import Column

class UnsupportedBuffers(TypeError):pass

TREE_FIELDS=('leaf','values','feature','cut','missing','left','right')
def address(value):
    # Never dispatch to an instance/subclass's overridable buffer_info method.
    return ctypes.c_void_p(array.buffer_info(value)[0])

def columns_view(columns):
    if type(columns) not in (list,tuple) or not columns:raise ValueError('Nonempty column sequence required')
    if any(type(c) not in (array,Column) for c in columns):raise UnsupportedBuffers('Exact owned array/Column required')
    rows=array.__len__(columns[0])
    if any(c.typecode not in ('B','H','I') or c.itemsize not in (1,2,4) or array.__len__(c)!=rows for c in columns):raise UnsupportedBuffers('Aligned owned B/H/I columns required')
    widths=array('I',(c.itemsize for c in columns));nominal=array('B',(bool(getattr(c,'nominal',False)) for c in columns))
    pointers=(ctypes.c_void_p*len(columns))(*(array.buffer_info(c)[0] for c in columns))
    return dict(rows=rows,widths=widths,nominal=nominal,pointers=pointers,owners=(*columns,widths,nominal))

def tree_key(nodes):
    if type(nodes) not in (list,tuple):raise UnsupportedBuffers('Exact owned node list/tuple required')
    if not nodes:raise ValueError('Nonempty declarative tree required')
    # Include type and all numeric bits. True/1,-0/+0 and NaN payloads must not
    # accidentally alias a previously checked layout. A changed tuple/list or
    # even an intentional frozen-dataclass bypass produces a different key.
    key=[]
    for n in nodes:
        if type(n.is_leaf) is not bool or type(n.missing_left) is not bool:raise ValueError('Typed tree flags required')
        ints=(n.feature,n.threshold_bin,n.left,n.right)
        if any(type(v) is not int or not -(2**63)<=v<2**63 for v in ints):raise ValueError('Int64 tree fields required')
        key.append((n.is_leaf,struct.pack('=d',n.value),*ints,n.missing_left))
    return tuple(key)

def pack_tree(key):
    leaf=array('B');values=array('d');feature=array('q');cut=array('q');missing=array('B');left=array('q');right=array('q')
    for is_leaf,value,f,c,l,r,m in key:
        leaf.append(is_leaf);values.append(struct.unpack('=d',value)[0]);feature.append(f);cut.append(c);left.append(l);right.append(r);missing.append(m)
    return dict(zip(TREE_FIELDS,(leaf,values,feature,cut,missing,left,right)))

def owned_bytes(key,packed):
    # Conservative accounting includes key containers and scalar references,
    # even when CPython interns/shares them. No input columns/models retained.
    return (sys.getsizeof(key)+sum(sys.getsizeof(row)+sum(sys.getsizeof(v) for v in row) for row in key)
            +sys.getsizeof(packed)+sum(sys.getsizeof(v) for v in packed.values())+256)

class TreeCache:
    def __init__(self,max_bytes=4*1024**2):
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('Nonnegative cache byte allowance')
        self.max_bytes=max_bytes;self.bytes=0;self.hits=0;self.misses=0;self.evictions=0;self._data=OrderedDict();self._lock=threading.RLock()
    def prepare(self,nodes):
        key=tree_key(nodes)
        with self._lock:
            if key in self._data:
                self.hits+=1;packed,size=self._data.pop(key);self._data[key]=(packed,size);return packed
            self.misses+=1;packed=pack_tree(key);size=owned_bytes(key,packed)
            if size<=self.max_bytes:
                while self._data and self.bytes+size>self.max_bytes:
                    _,(_,old)=self._data.popitem(last=False);self.bytes-=old;self.evictions+=1
                self._data[key]=(packed,size);self.bytes+=size
            return packed
    def clear(self):
        with self._lock:self._data.clear();self.bytes=0
    def report(self):
        with self._lock:return dict(bytes_accounted=self.bytes,max_bytes=self.max_bytes,entries=len(self._data),hits=self.hits,misses=self.misses,evictions=self.evictions,scope='Conservative Python object accounting;not measured RSS')

def histogram_view(column,indices,g,h,r,u,size):
    if type(size) is not int or not 1<=size<=sys.maxsize//40:raise ValueError('Safe positive bin count required')
    if type(column) not in (array,Column) or column.typecode not in ('B','H','I') or column.itemsize not in (1,2,4):raise UnsupportedBuffers('Exact owned bin array/Column required')
    values=(g,h,r,u)
    if any(type(v) is not array or v.typecode!='d' or v.itemsize!=8 or array.__len__(v)!=array.__len__(column) for v in values):raise UnsupportedBuffers('Exact aligned binary64 arrays required')
    copied=not (type(indices) is array and indices.typecode=='Q' and indices.itemsize==8)
    if copied:
        if type(indices) not in (list,tuple,range,array):raise UnsupportedBuffers('Exact bounded node indices required')
        ids=array('Q')
        for i in indices:
            if type(i) is not int or not 0<=i<array.__len__(column):raise ValueError('Invalid row index')
            ids.append(i)
    else:ids=indices
    return dict(ids=ids,owners=(column,ids,*values),copy_bytes=len(ids)*8 if copied else 0)
