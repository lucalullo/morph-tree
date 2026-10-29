"""Integrated native execution with unchanged public estimator signatures."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
from .scalable_backend import ScalableBackend
from ._native.buffers import UnsupportedBuffers

_ACTIVE=ContextVar('morphtree_native_lab_backend',default=None);_PYTHON=object()


class NativeSession:
    def __init__(self,*,histogram_only=False):
        from ._native.loader import get_runtime,runtime_info
        self.kernel=get_runtime();self.info=runtime_info();self.binary_sha256=self.info['binary_sha256'];self.histogram_only=histogram_only
        self.calls=0;self.fallback_calls=0;self.indices_copy_bytes=0;self.seconds=0.;self.traversal_calls=0;self.small_batch_python_calls=0;self.traversal_fallback_calls=0
    def create(self,nominal_sizes):return (NativeHistogramBackend if self.histogram_only else NativeBackend)(self,nominal_sizes)
    def report(self):
        return dict(backend='MORPHTREE_INTEGRATED_NATIVE_ABI1',binary_sha256=self.binary_sha256,
                    histogram_calls=self.calls,fallback_calls=self.fallback_calls,
                    indices_copy_bytes=self.indices_copy_bytes,histogram_seconds_including_bridge=self.seconds,
                    traversal_calls=self.traversal_calls,small_batch_python_calls=self.small_batch_python_calls,
                    traversal_fallback_calls=self.traversal_fallback_calls,
                    scope='Current process execution telemetry,not restored historical fit cost',runtime={**self.info,'cache':self.kernel.trees.report()})


class NativeHistogramBackend(ScalableBackend):
    name='morphtree-native-histogram'
    def __init__(self,session,nominal_sizes=None):super().__init__(nominal_sizes);self.session=session
    def histogram(self,column,indices,g,h,r,u,size):
        tick=time.perf_counter()
        try:
            result,copied=self.session.kernel.histogram(column,indices,g,h,r,u,size)
        except UnsupportedBuffers:
            self.session.fallback_calls+=1
            return super().histogram(column,indices,g,h,r,u,size)
        # Native rejection/error is not a fallback:it must stop qualification.
        self.session.calls+=1;self.session.seconds+=time.perf_counter()-tick
        self.session.indices_copy_bytes+=copied
        return result


class NativeBackend(NativeHistogramBackend):
    name='morphtree-integrated-native'
    def predict_tree(self,nodes,columns):
        # INTEGRATED013:Python wins at1/8/32rows,C wins from128rows.
        # Equivalent execution route only;no row limit or sampling.
        if columns and len(columns[0])<128:
            self.session.small_batch_python_calls+=1;return ScalableBackend.predict_tree(self,nodes,columns)
        try:result=self.session.kernel.predict_tree(nodes,columns)
        except UnsupportedBuffers:
            self.session.traversal_fallback_calls+=1;return ScalableBackend.predict_tree(self,nodes,columns)
        self.session.traversal_calls+=1;return result

def bind_backend():
    """One provider captured per fit/load/export;verified native use is automatic."""
    active=_ACTIVE.get()
    if active is _PYTHON:return ScalableBackend
    return (NativeSession() if active is None else active).create


@contextmanager
def _scope(value):
    if _ACTIVE.get() is not None:raise RuntimeError('Nested native backend scopes are not supported')
    token=_ACTIVE.set(value)
    try:yield value
    finally:_ACTIVE.reset(token)

@contextmanager
def python_backend_scope():
    """Explicit reference route for qualification,not a public estimator knob."""
    with _scope(_PYTHON):yield

@contextmanager
def native_histogram_scope():
    with _scope(NativeSession(histogram_only=True)) as value:yield value

@contextmanager
def native_backend_scope():
    with _scope(NativeSession()) as value:yield value
