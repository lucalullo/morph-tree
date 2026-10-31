"""Public NumPy output boundary; no learning, clipping or renormalization."""

def prediction_array(values, *, regression=False):
    import numpy as np
    result=np.asarray(values,dtype=np.float64 if regression else object if len(values) and isinstance(values[0],str) else None)
    if result.ndim!=1:raise ValueError('One-dimensional scalar predictions required')
    if not regression and any((v.item() if hasattr(v,'item') else v)!=old for v,old in zip(result,values)):
        # Never round large/mixed integer class IDs to binary64 or stringify them.
        result=np.asarray(values,dtype=object)
    if regression and not np.isfinite(result).all():raise ValueError('Finite regression predictions required')
    return result

def probability_array(values, n_classes, *, n_samples=None, check=None):
    import numpy as np
    result=np.asarray(values,dtype=np.float64)
    if result.ndim!=2 or result.shape[1]!=n_classes or n_classes<2 or n_samples is not None and result.shape[0]!=n_samples:
        raise ValueError('Probability shape must be (n_samples,n_classes)')
    for start in range(0,len(result),4096):
        if check:check()
        block=result[start:start+4096]
        if not np.isfinite(block).all() or np.any(block<0) or np.any(block>1):raise ValueError('Probabilities must be finite and in [0,1]')
        if not np.allclose(block.sum(axis=1),1.,rtol=0.,atol=1e-10):raise ValueError('Probability rows must sum to one')
    if check:check()
    return result

def numpy_export_source(source, classes, regression):
    """Wrap explicit NumPy exports while retaining the standalone list default."""
    import inspect
    extra='\n# Explicit NumPy output adapter; requires NumPy, not morphtree.\n'
    extra+='OUTPUT_FORMAT = "numpy"\nclasses_ = '+repr(list(classes))+'\n'
    extra+=inspect.getsource(prediction_array)+'\n'+inspect.getsource(probability_array)
    extra+='\n_morphtree_python_predict = predict\ndef predict(X):\n    return prediction_array(_morphtree_python_predict(X), regression='+repr(regression)+')\n'
    if not regression:
        extra+='\n_morphtree_python_predict_proba = predict_proba\ndef predict_proba(X):\n    return probability_array(_morphtree_python_predict_proba(X), len(classes_), n_samples=len(X))\n'
    return source+extra
