"""Public2.0 call contracts; historical weights keep their original decoder."""
from pathlib import Path
import os


def validate_workdir(value):
    if value is None:return None
    if not isinstance(value,(str,os.PathLike)):
        raise ValueError('Existing local working directory required')
    value=os.fspath(value)
    if not isinstance(value,str) or not value:raise ValueError('Existing local working directory required')
    from ._experimental.admission import safe_path
    path=Path(value)
    # safe_path rejects links/reparse ancestors before resolving them.
    checked=safe_path(path.absolute(),path.absolute())
    if not checked.is_dir():raise ValueError('Existing local working directory required')
    return checked


class HistoricalModel:
    """A real subclass of the new facade with unchanged legacy learned state.

    Public inspection/IO use the original decoder/engine. A new fit uses the
    scalable controller and only replaces that historical state on success.
    """
    _historical_delegates=frozenset(('predict','predict_proba','score','save',
        'report','get_search_report','get_algorithm_program','get_recipe',
        'get_mutation_history','cost_report','available_variants','save_recipe',
        'export_python','export_standalone','export_algorithm','task_','classes_',
        'n_features_in_','feature_names_in_','best_family_','best_params_',
        'stop_reason_','_engine','_loaded_workspace'))

    def __getattribute__(self,name):
        state=object.__getattribute__(self,'__dict__')
        legacy=state.get('_historical_model')
        if legacy is not None and name in HistoricalModel._historical_delegates:
            if name=='_loaded_workspace':return getattr(legacy._engine,name)
            return getattr(legacy,name)
        return super().__getattribute__(name)

    def __sklearn_is_fitted__(self):
        return '_historical_model' in self.__dict__ or super().__sklearn_is_fitted__()

    def fit(self,X,y):
        historical=self.__dict__.pop('_historical_model',None)
        try:return super().fit(X,y)
        except BaseException:
            if historical is not None:self.__dict__['_historical_model']=historical
            raise


def wrap_historical(legacy,facade,memory_budget):
    from .scalable import MorphTreeClassifier,MorphTreeRegressor
    if not hasattr(legacy,'_engine'):
        from . import estimators
        kind=estimators.MorphTreeRegressor if legacy.task_=='regression' else estimators.MorphTreeClassifier
        wrapped=kind.__new__(kind);wrapped._engine=legacy;legacy=wrapped
    kind=MorphTreeRegressor if legacy.task_=='regression' else MorphTreeClassifier
    params=legacy.get_params()
    params.update(memory_budget=memory_budget)
    # This is an ordinary Python subclass, not __class__ spoofing or a rewrite
    # of weights. get_params describes its next scalable refit, not old training.
    cls=type('Historical'+kind.__name__,(HistoricalModel,kind),{'__module__':__name__})
    obj=cls(**params);obj._historical_model=legacy
    return obj
