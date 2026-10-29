"""One native scalar boosting loop for binary logloss and squared error."""
from __future__ import annotations

from array import array
import math
import random

from .backend import PythonBackend
from .data import Binner
from .config import AlgorithmGenome, validate_task_genome
from .metrics import logloss, sigmoid, rmse, multiclass_metrics
from .tree import HistogramTree


class TrainingCache:
    """One fit-call cache. Never shared with other datasets or a full-data refit."""
    def __init__(self, X_train, X_search, backend):
        self.X_train, self.X_search, self.backend = X_train, X_search, backend
        self.entries = {}

    def get(self, genome, params):
        key = (genome.binning_policy, params.max_bins)
        if key not in self.entries:
            binner = Binner(params.max_bins, genome.binning_policy).fit(self.X_train)
            self.entries[key] = (binner, self.backend.bin(binner, self.X_train), self.backend.bin(binner, self.X_search))
        return self.entries[key]


class BinaryHistogramBooster:
    """Historical private name retained; regression uses this same tree/fit loop."""
    def __init__(self, genome, params, seed=42, backend=None, *, memory_enabled=True, collect_diagnostics=False,
                 objective='binary_logloss', n_classes=None):
        if objective not in ('binary_logloss', 'squared_error', 'softmax', 'ovr'):
            raise ValueError('Unsupported boosting objective')
        validate_task_genome('binary' if objective=='binary_logloss' else 'regression' if objective=='squared_error' else 'multiclass',genome)
        if objective != 'binary_logloss' and (memory_enabled or collect_diagnostics):
            raise ValueError('New tasks support fixed Base without memory or discovery')
        if objective in ('softmax', 'ovr') and (type(n_classes) is not int or n_classes < 2):
            raise ValueError('Multiclass requires an integer class dimension of at least two')
        self.n_classes = n_classes
        self.objective = objective
        self.memory_enabled = memory_enabled
        self.collect_diagnostics = collect_diagnostics
        self.genome, self.params, self.seed = genome, params, seed
        self.backend = backend or PythonBackend()
        self.binner = None
        self.trees, self.steps = [], []
        self.base_score = 0.0
        self.n_features_ = 0
        self.memory_ = None
        self.search_curve_ = []
        self.stopped_early_ = False

    def fit(self, X, y, *, eval_set=None, callback=None, cache=None, early_stopping_rounds=0, prepared_data=None):
        if prepared_data is not None and (cache is not None or eval_set is not None):
            raise ValueError('Prepared training data cannot be mixed with another cache/evaluation scope')
        if self.n_classes is not None:
            return self._fit_multiclass(X, y, eval_set, callback, early_stopping_rounds, prepared_data)
        self.n_features_ = len(X[0])
        if self.objective == 'squared_error':
            self.base_score = y[0] if all(v == y[0] for v in y) else math.fsum(v / len(y) for v in y)
        else:
            prior = min(1 - 1e-12, max(1e-12, sum(y) / len(y)))
            self.base_score = math.log(prior / (1 - prior))
        if prepared_data is not None:
            self.binner, Xb = prepared_data.get(X, self.genome, self.params, self.backend)
            eval_bins = None
        elif cache is not None:
            if cache.X_train is not X or eval_set is None or cache.X_search is not eval_set[0]:
                raise ValueError('Training cache belongs to different input objects')
            self.binner, Xb, eval_bins = cache.get(self.genome, self.params)
        else:
            self.binner = Binner(self.params.max_bins, self.genome.binning_policy).fit(X)
            Xb = self.backend.bin(self.binner, X)
            eval_bins = self.backend.bin(self.binner, eval_set[0]) if eval_set else None
        sizes = [len(c) + 2 for c in self.binner.thresholds]
        rng = random.Random(self.seed)
        raw = [self.base_score] * len(y)
        eval_raw = [self.base_score] * len(eval_set[1]) if eval_set else None
        needs_memory = (self.memory_enabled or self.collect_diagnostics or self.genome.error_policy != "none"
                        or self.genome.sample_policy != "uniform" or self.genome.leaf_policy == "uncertainty"
                        or "UNCERTAINTY_GAIN" in self.genome.split_program)
        if needs_memory:
            from .diagnostics import ErrorMemory
            self.memory_ = ErrorMemory(len(y), self.params.error_memory_beta)
        else:
            self.memory_ = None
        no_uncertainty = array("d", [0.0]) * len(y)
        self.trees, self.steps, self.search_curve_ = [], [], []
        self.stopped_early_ = False
        sample_count = self.params.row_sample_count(len(y))
        best_loss, best_count, stale = math.inf, 0, 0
        for iteration in range(self.params.n_estimators):
            p, gradient, hessian = (self.backend.regression_gradients(raw, y) if self.objective == 'squared_error'
                                    else self.backend.gradients(raw, y))
            if needs_memory:
                if not self.memory_enabled:
                    self.memory_ = ErrorMemory(len(y), self.params.error_memory_beta)
                self.memory_.update(y, p)
            residual = array("d", (label - probability for label, probability in zip(y, p)))
            uncertainty = self.memory_.uncertainty if needs_memory else no_uncertainty
            if self.genome.error_policy == "persistent_weighting":
                for i, hardness in enumerate(self.memory_.hardness):
                    weight = 1 + self.params.error_weight * hardness
                    gradient[i] *= weight
                    hessian[i] *= weight
            ids = None
            if sample_count < len(y):
                count = sample_count
                if self.genome.sample_policy == "uniform":
                    ids = sorted(rng.sample(range(len(y)), count))
                else:
                    weights = self.memory_.hardness if self.genome.sample_policy == "hardness" else self.memory_.uncertainty
                    keys = [(-math.log(max(rng.random(), 1e-15)) / max(w, 1e-6), i) for i, w in enumerate(weights)]
                    ids = sorted(i for _, i in sorted(keys)[:count])
            tree = HistogramTree(self.genome, self.params, self.backend).fit(Xb, gradient, hessian, residual, uncertainty,
                                                                                 rng, indices=ids, sizes=sizes)
            step = self.params.learning_rate
            if self.genome.update_policy == "decay":
                step /= 1 + self.params.update_decay * iteration / max(self.params.n_estimators - 1, 1)
            raw = self.backend.add_update(raw, step, tree.predict_binned(Xb))
            self.trees.append(tree)
            self.steps.append(step)
            loss = None
            if eval_set:
                eval_raw = self.backend.add_update(eval_raw, step, tree.predict_binned(eval_bins))
                loss = (rmse(eval_set[1], eval_raw) if self.objective == 'squared_error'
                        else logloss(eval_set[1], self.backend.probabilities(eval_raw)))
                self.search_curve_.append(loss)
                if loss < best_loss - (0.0 if self.objective == 'squared_error' else 1e-12):
                    best_loss, best_count, stale = loss, len(self.trees), 0
                else:
                    stale += 1
            if callback is not None and callback(iteration + 1, loss) is False:
                self.stopped_early_ = True
                break
            if early_stopping_rounds and eval_set and stale >= early_stopping_rounds:
                self.stopped_early_ = True
                break
        if early_stopping_rounds and best_count and best_count < len(self.trees):
            self.trees = self.trees[:best_count]
            self.steps = self.steps[:best_count]
            # Rebuild memory so it describes the retained ensemble, not discarded trees.
            raw = [self.base_score] * len(y)
            if needs_memory:
                self.memory_ = ErrorMemory(len(y), self.params.error_memory_beta)
            for tree, step in zip(self.trees, self.steps):
                if needs_memory:
                    self.memory_.update(y, self.backend.probabilities(raw))
                raw = self.backend.add_update(raw, step, tree.predict_binned(Xb))
        if needs_memory:
            self.memory_.update(y, self.backend.probabilities(raw))
        if any(not math.isfinite(float(v)) for v in raw):
            raise ArithmeticError("Training produced nonfinite predictions")
        return self

    def _fit_multiclass(self, X, y, eval_set, callback, early_stopping_rounds, prepared_data=None):
        """K scalar trees per round; one binner/tree builder and O(n*K) scores.

        Derivatives of every class use the same frozen round probabilities.
        OvR optimizes K binary losses; only its output is simplex-normalized.
        """
        k = self.n_classes
        self.n_features_ = len(X[0])
        counts = [0]*k
        for label in y:
            counts[label] += 1
        priors = [min(1-1e-12, max(1e-12, count/len(y))) for count in counts]
        self.base_score = [math.log(p) if self.objective == 'softmax' else math.log(p/(1-p)) for p in priors]
        if self.objective == 'softmax':
            self.backend.center_multiclass([self.base_score])
        if prepared_data is None:
            self.binner = Binner(self.params.max_bins, self.genome.binning_policy).fit(X)
            bins = self.backend.bin(self.binner, X)
        else:
            self.binner, bins = prepared_data.get(X, self.genome, self.params, self.backend)
        eval_bins = self.backend.bin(self.binner, eval_set[0]) if eval_set else None
        sizes = [len(c)+2 for c in self.binner.thresholds]
        raw = self.backend.multiclass_raw(self.base_score, len(y))
        eval_raw = self.backend.multiclass_raw(self.base_score, len(eval_set[1])) if eval_set else None
        rng = random.Random(self.seed)
        zeros = array('d', [0.])*len(y)
        self.trees, self.steps, self.search_curve_ = [], [], []
        self.stopped_early_ = False
        best, best_count, stale = math.inf, 0, 0
        count = self.params.row_sample_count(len(y))
        for iteration in range(self.params.n_estimators):
            probabilities = self.backend.multiclass_probabilities(raw, self.objective, normalize=False)
            for column in range(k):
                if callback is not None and callback(iteration, None) is False:
                    # A partial round must never become a fitted/serialized model.
                    raise RuntimeError('Multiclass fit interrupted during a round')
                gradient, hessian = self.backend.multiclass_gradient_column(probabilities, y, column, self.objective)
                residual = array('d', (-v for v in gradient))
                ids = sorted(rng.sample(range(len(y)), count)) if count < len(y) else None
                tree = HistogramTree(self.genome, self.params, self.backend).fit(
                    bins, gradient, hessian, residual, zeros, rng, indices=ids, sizes=sizes)
                step = self.params.learning_rate
                self.backend.multiclass_update(raw, column, step, tree.predict_binned(bins))
                if eval_set:
                    self.backend.multiclass_update(eval_raw, column, step, tree.predict_binned(eval_bins))
                self.trees.append(tree)
                self.steps.append(step)
            if self.objective == 'softmax':
                self.backend.center_multiclass(raw)
                if eval_set:
                    self.backend.center_multiclass(eval_raw)
            loss = None
            if eval_set:
                loss = multiclass_metrics(eval_set[1], self.backend.multiclass_probabilities(eval_raw, self.objective))['logloss']
                self.search_curve_.append(loss)
                if loss < best:
                    best, best_count, stale = loss, len(self.trees), 0
                else:
                    stale += 1
            if callback is not None and callback(iteration+1, loss) is False:
                self.stopped_early_ = True
                break
            if early_stopping_rounds and eval_set and stale >= early_stopping_rounds:
                self.stopped_early_ = True
                break
        if early_stopping_rounds and best_count:
            self.trees, self.steps = self.trees[:best_count], self.steps[:best_count]
        if any(not math.isfinite(float(v)) for row in raw for v in row):
            raise ArithmeticError('Multiclass training produced nonfinite scores')
        return self

    def predict_raw(self, X):
        if self.binner is None:
            raise RuntimeError("Booster is not fitted")
        Xb = self.backend.bin(self.binner, X)
        if self.n_classes is not None:
            raw = self.backend.multiclass_raw(self.base_score, len(X))
            for index, (tree, step) in enumerate(zip(self.trees, self.steps)):
                column = index % self.n_classes
                self.backend.multiclass_update(raw, column, step, tree.predict_binned(Xb))
                if column == self.n_classes-1 and self.objective == 'softmax':
                    self.backend.center_multiclass(raw)
            return raw
        raw = [self.base_score] * len(X)
        for tree, step in zip(self.trees, self.steps):
            raw = self.backend.add_update(raw, step, tree.predict_binned(Xb))
        return raw

    def predict_proba(self, X):
        if self.n_classes is not None:
            return self.backend.multiclass_probabilities(self.predict_raw(X), self.objective)
        return self.backend.probabilities(self.predict_raw(X))

    def to_dict(self):
        return {"base_score": self.base_score, "n_features": self.n_features_, "binner": self.binner.to_dict(),
                "steps": self.steps, "trees": [tree.to_dict() for tree in self.trees]}
