"""Integrated ABI execution;immutable keys,per-call buffers and retained GIL."""
from array import array
from contextlib import ExitStack
import ctypes
from .buffers import TreeCache,columns_view,histogram_view,TREE_FIELDS,address,UnsupportedBuffers
from ..scalable_backend import Column

class Runtime:
    def __init__(self,histogram_library,traversal_library=None,*,cache_bytes=4*1024**2):
        # The loader,not a model packet,chooses and verifies these handles.
        self.histogram_library=histogram_library;self.traversal_library=traversal_library or histogram_library
        self.histogram_library.mt_histogram_abi_version.restype=ctypes.c_uint32
        self.traversal_library.mt_traversal_abi_version.restype=ctypes.c_uint32
        if self.histogram_library.mt_histogram_abi_version()!=1 or self.traversal_library.mt_traversal_abi_version()!=1:raise ImportError('Unsupported MorphTree native ABI')
        f=self.histogram_library.mt_histogram_f64_v1;f.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64,*([ctypes.c_void_p]*4),ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64];f.restype=ctypes.c_int;self.histogram_function=f
        f=self.traversal_library.mt_predict_tree_f64_v1;f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_uint64]*2+[ctypes.c_void_p]*7+[ctypes.c_uint64,ctypes.c_void_p,ctypes.c_uint64];f.restype=ctypes.c_int;self.traversal_function=f
        self.trees=TreeCache(cache_bytes)
    def histogram(self,column,indices,g,h,r,u,size):
        # Pin caller-owned arrays before validating lengths or running any
        # conversion. Concurrent resize must not invalidate a checked shape.
        if type(column) not in (array,Column) or any(type(v) is not array for v in (g,h,r,u)):
            raise UnsupportedBuffers('Exact owned bin and statistic arrays required')
        with ExitStack() as pins:
            for value in (column,g,h,r,u):pins.enter_context(memoryview(value))
            if type(indices) is array:pins.enter_context(memoryview(indices))
            p=histogram_view(column,indices,g,h,r,u,size);owners=p['owners']
            pins.enter_context(memoryview(p['ids']))
            output=array('d',[0.])*(5*size)
            pointers=[address(v) for v in owners]
            code=self.histogram_function(pointers[0],column.itemsize,array.__len__(column),pointers[1],array.__len__(p['ids']),*pointers[2:],size,address(output),len(output))
            if code:raise ValueError('Native histogram rejected request/status='+str(code))
            from ..scalable_backend import Hist
            return Hist([output[i*size:(i+1)*size] for i in range(5)],getattr(column,'nominal',False)),p['copy_bytes']
    def predict_tree(self,nodes,columns):
        if type(columns) not in (list,tuple) or not columns:raise ValueError('Nonempty column sequence required')
        columns=tuple(columns)
        if any(type(c) not in (array,Column) for c in columns):raise UnsupportedBuffers('Exact owned array/Column required')
        # Snapshot the container and pin columns before nominal.__bool__ or
        # a node value's numeric conversion can execute Python callbacks.
        with ExitStack() as pins:
            for value in columns:pins.enter_context(memoryview(value))
            c=columns_view(columns);p=self.trees.prepare(nodes)
            for value in (c['widths'],c['nominal'],*(p[k] for k in TREE_FIELDS)):
                pins.enter_context(memoryview(value))
            output=array('d',[0.])*c['rows'];pointer=address
            code=self.traversal_function(c['pointers'],pointer(c['widths']),pointer(c['nominal']),len(columns),c['rows'],*(pointer(p[k]) for k in TREE_FIELDS),array.__len__(p['leaf']),pointer(output),len(output))
            if code:raise ValueError('Native traversal rejected request/status='+str(code))
            return list(output)
