"""Optional sklearn array boundary; the proprietary learner stays Pure Python.

The underlying facade keeps its list outputs and independent public API.
This view delegates fitting to that same facade and adds no training, tuning,
sampling or preprocessing. sklearn is optional; NumPy is used only for arrays.
"""
from .scalable import _Scalable
from .scalable_data import Input


def _labels(values):
    import numpy as np
    result=np.asarray(values,dtype=object if values and isinstance(values[0],str) else None)
    # In particular do not round mixed large integers to binary64 merely to
    # satisfy an external scorer. Such scorers may reject object labels; the
    # original facade supports them and this view must keep their exact values.
    if any((v.item() if hasattr(v,'item') else v)!=old for v,old in zip(result,values)):
        result=np.asarray(values,dtype=object)
    return result


class _ClassifierMethod:
    """Keep the public response-method name while hiding it for regression."""
    def __init__(self,function):self.function=function
    def __get__(self,instance,owner):
        if instance is None:return self.function
        if instance.estimator.task!='classification':raise AttributeError('Regression has no probabilities')
        return self.function.__get__(instance,owner)


class SklearnView:
    """Explicit array-output adapter with estimator__parameter configuration.

    Qualify external tools separately; sample weights, sparse data, metadata
    routing and all estimator-check contracts are not implied by this view.
    Saving via estimator.save() persists the native model; restore it and call
    as_sklearn() again. Cloning always produces an unfitted native estimator.
    """
    def __init__(self,estimator):
        if not isinstance(estimator,_Scalable) or estimator.task not in ('classification','regression'):
            raise TypeError('A native MorphTree task facade is required')
        self.estimator=estimator
    def get_params(self,deep=True):
        params={'estimator':self.estimator}
        if deep:params.update({'estimator__'+k:v for k,v in self.estimator.get_params(deep=True).items()})
        return params
    def set_params(self,**values):
        if not values:return self
        candidate=values.get('estimator',self.estimator)
        type(self)(candidate)
        unknown=[k for k in values if k!='estimator' and not k.startswith('estimator__')]
        if unknown:raise ValueError('Unknown view parameter: '+', '.join(sorted(unknown)))
        nested={k[len('estimator__'):]:v for k,v in values.items() if k!='estimator'}
        if nested:candidate.set_params(**nested)
        self.estimator=candidate
        return self
    @property
    def _estimator_type(self):return self.estimator._estimator_type
    def __sklearn_tags__(self):return self.estimator.__sklearn_tags__()
    def __sklearn_is_fitted__(self):return self.estimator.__sklearn_is_fitted__()
    @property
    def classes_(self):
        if self.estimator.task!='classification':raise AttributeError('Regression has no classes')
        self.estimator._resources().reserve(len(self.estimator.classes_)*64,'Sklearn class-label array')
        return _labels(self.estimator.classes_)
    @property
    def n_features_in_(self):return self.estimator.n_features_in_
    def fit(self,X,y):
        self.estimator.fit(X,y)
        return self
    def predict(self,X):
        if not hasattr(self.estimator,'model_'):raise RuntimeError('Not fitted')
        resources=self.estimator._resources()
        resources.reserve(Input(X).n*64,'Native prediction list plus sklearn output array')
        values=self.estimator.predict(X);resources.check()
        result=_labels(values);resources.check();return result
    @_ClassifierMethod
    def predict_proba(self,X):
        import numpy as np
        if not hasattr(self.estimator,'model_'):raise RuntimeError('Not fitted')
        resources=self.estimator._resources()
        resources.reserve(Input(X).n*(len(self.estimator.classes_)*48+64),'Native probability rows plus sklearn output array')
        values=self.estimator.predict_proba(X);resources.check()
        result=np.asarray(values,dtype=float);resources.check();return result
    def score(self,X,y):return self.estimator.score(X,y)
