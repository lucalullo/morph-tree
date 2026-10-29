"""Known2.0artifact compatibility,never fallback from a malformed new model.

Original2.0inference/serialization implementation and original input limits are
preserved for these historical artifacts. Newly trained scalable models retain
the current automatic batch/resource paths. No old weights/program are rewritten
or reinterpreted as a newly trained model.
"""
PUBLIC_2_0_SCHEMA='product006-native-control-v1'


def restore_public_2_0(path,sha256,intent,*,workdir=None):
    # Use the exact retained decoder,including variant/provenance/graph checks.
    # A checksum of the already validated immutable snapshot prevents re-opening
    # another file under a replaced path without detecting the changed bytes.
    if intent=='classification':
        from .estimators import MorphTreeClassifier
        result=MorphTreeClassifier.load(path,sha256=sha256,workdir=workdir)
    elif intent=='regression':
        from .estimators import MorphTreeRegressor
        result=MorphTreeRegressor.load(path,sha256=sha256,workdir=workdir)
    else:
        from .api import MorphTree
        result=MorphTree.load(path,sha256=sha256,workdir=workdir)
    # The legacy engine must retain its cache for its complete lifetime. Close
    # it explicitly at the same owner-collection boundary instead of relying
    # on TemporaryDirectory's warning-producing implicit cleanup. The finalizer
    # captures only the cache,never the engine,so it cannot keep the owner alive.
    import weakref
    owner=getattr(result,'_engine',result)
    workspace=getattr(owner,'_loaded_workspace',None)
    if workspace is not None:weakref.finalize(owner,workspace.cleanup)
    return result
