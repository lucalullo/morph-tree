"""Task-specific public estimators sharing the existing evolutionary engine."""
import copy

from .api import MorphTree


def _score_targets(y, length):
    from ._experimental.preprocessing import _m26_scalar
    values = y.tolist() if hasattr(y, 'tolist') else y
    if not isinstance(values, (list, tuple)) or not values or len(values) != length:
        raise ValueError('score requires nonempty aligned scalar targets')
    values = [_m26_scalar(value) for value in values]
    if any(value is None for value in values):
        raise ValueError('Missing score targets are unsupported')
    return values


class _TaskEstimator:
    _task_kind = None
    _fitted_attributes = frozenset(('task_', 'n_features_in_', 'feature_names_in_',
        'best_family_', 'best_params_', 'stop_reason_', 'last_attempt_'))

    def __init__(self, *, random_state=None, verbosity=0, use_optuna=True,
                 max_mutations=2, time_budget='auto', n_trials='auto', n_jobs=1,
                 capacity=None, mode='adaptive', operators=None):
        self._engine = MorphTree(task=self._task_kind, random_state=random_state,
            verbosity=verbosity, use_optuna=use_optuna, max_mutations=max_mutations,
            time_budget=time_budget, n_trials=n_trials, n_jobs=n_jobs,
            capacity=capacity, mode=mode, operators=operators)

    def get_params(self, deep=True):
        engine = self._engine
        return copy.deepcopy(dict(random_state=engine.random_state,
            verbosity=engine.verbose, use_optuna=engine.tuning != 'none',
            max_mutations=engine._max_mutations, time_budget=engine.time_budget,
            n_trials=engine.n_trials, n_jobs=engine.n_jobs, capacity=engine.capacity,
            mode=engine.mode,
            operators=('growth_policy:depthwise',) if engine._research_growth else None))

    def set_params(self, **values):
        params = self.get_params()
        if set(values) - set(params):
            raise ValueError('Unknown estimator parameter: ' + ', '.join(sorted(set(values)-set(params))))
        replacement = type(self)(**(params | values))
        self._engine = replacement._engine
        return self

    def __getattr__(self, name):
        # Forward only documented attributes; do not expose the internal engine
        # as if the task facade itself were a raw native learner.
        if name in self._fitted_attributes or (name == 'classes_' and self._task_kind == 'classification'):
            return copy.deepcopy(getattr(self._engine, name))
        if name in ('random_state', 'verbosity', 'use_optuna', 'max_mutations',
                    'time_budget', 'n_trials', 'n_jobs', 'capacity', 'mode', 'operators'):
            return self.get_params()[name]
        raise AttributeError(name)

    def fit(self, X, y):
        # Input materialization, context/sealed-data guards, finite budgets and
        # fit rollback remain in the same engine as the legacy MorphTree API.
        self._engine.fit(X, y)
        return self

    def predict(self, X):
        return self._engine.predict(X)

    def save(self, path, *, variant='all'):
        return self._engine.save(path, variant=variant)

    @classmethod
    def load(cls, path, *, sha256=None, workdir=None):
        engine = MorphTree.load(path, sha256=sha256, workdir=workdir)
        allowed = ('binary', 'multiclass') if cls._task_kind == 'classification' else ('regression',)
        if engine.task_ not in allowed:
            workspace = getattr(engine, '_loaded_workspace', None)
            if workspace is not None:
                workspace.cleanup()
            raise ValueError('Saved model task is incompatible with ' + cls.__name__)
        # Keep loaded weights, encoder, variants, recipe and evidence unchanged.
        # A classifier refit may switch between binary and multiclass targets.
        engine.task = cls._task_kind
        obj = cls.__new__(cls)
        obj._engine = engine
        return obj

    def report(self):
        return self._engine.report()

    def get_search_report(self):
        return self._engine.get_search_report()

    def get_algorithm_program(self):
        return self._engine.get_algorithm_program()

    def get_mutation_history(self):
        return self._engine.get_mutation_history()

    def get_recipe(self):
        return self._engine.get_recipe()

    def cost_report(self):
        return self._engine.cost_report()

    def available_variants(self):
        return self._engine.available_variants()

    def save_recipe(self, path, *, variant='selected'):
        return self._engine.save_recipe(path, variant=variant)

    def export_python(self, path, *, variant='selected'):
        return self._engine.export_python(path, variant=variant)

    def export_standalone(self, path):
        return self._engine.export_standalone(path)

    def export_algorithm(self, path):
        return self._engine.export_algorithm(path)


class MorphTreeClassifier(_TaskEstimator):
    """Native evolutionary classification; resolves binary versus multiclass."""
    _task_kind = 'classification'
    _estimator_type = 'classifier'

    def predict_proba(self, X):
        return self._engine.predict_proba(X)

    def score(self, X, y):
        prediction = self.predict(X)
        targets = _score_targets(y, len(prediction))
        return sum(a == b for a, b in zip(targets, prediction)) / len(targets)


class MorphTreeRegressor(_TaskEstimator):
    """Native evolutionary regression with an explicitly numeric target."""
    _task_kind = 'regression'
    _estimator_type = 'regressor'

    def score(self, X, y):
        from .metrics import r2_score
        prediction = self.predict(X)
        return r2_score(_score_targets(y, len(prediction)), prediction)
