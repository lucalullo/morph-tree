"""Explicit/dynamic source metadata is distinct from pandas column/index data.

Only genuine inherited pandas attribute handlers qualify for the data-forwarding
exception. Static descriptors/instance/class markers and custom attribute
interceptors remain fail-closed. No optional dependency is imported here.
"""
import inspect,sys
MARKERS=('_morphtree_source_role','_morphtree_scientific_source','_morphtree_sealed')
ABSENT=object()

def controlled_source(obj):
    for name in MARKERS:
        if inspect.getattr_static(obj,name,ABSENT) is not ABSENT:return True
        value=getattr(obj,name,ABSENT)
        if value is ABSENT:continue
        pd=sys.modules.get('pandas');base=None
        if pd is not None:
            if isinstance(obj,pd.DataFrame):base=pd.DataFrame;axis='columns'
            elif isinstance(obj,pd.Series):base=pd.Series;axis='index'
        if base is not None:
            handlers=('__getattr__','__getattribute__')
            original=all(inspect.getattr_static(type(obj),method,ABSENT) is inspect.getattr_static(base,method,ABSENT) for method in handlers)
            if original and name in getattr(obj,axis):continue
        return True
    return False
