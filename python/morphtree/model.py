"""Native Base classifier; optional numeric tuning and experimental discovery."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import copy
import hashlib
import json
import math
import platform
import struct
import time

from .backend import PythonBackend
from .booster import BinaryHistogramBooster, TrainingCache
from .config import AlgorithmGenome, HyperParams, MorphTreeConfig, SPEC_VERSION, VERSION, validate_task_genome, decode_task_genome, TASK_GROWTH_FORMATS
from .data import get_feature_names, split_indices, regression_split_indices, subset, validate_names, validate_X, validate_xy, validate_y, validate_regression_y, validate_multiclass_y
from .metrics import accuracy, logloss, regression_metrics, r2_score, multiclass_metrics
from .persistence import LoadLimits, decode_booster, decode_multiclass_booster, decode_species, integer, read_json, require_keys, write_json
from .species import AlgorithmSpecies
from .control import RunControl, SearchInterrupted
from .public import public_config, imported_report, imported_regression_report, imported_multiclass_report, recipe_settings, RECIPE_FIELDS


@dataclass
class FitReport:
    data: dict

    def to_dict(self):
        return copy.deepcopy(self.data)


def dataset_fingerprint(splits, *, task='binary'):
    digest = hashlib.sha256()
    if task == 'regression':
        digest.update(b'MorphTree:regression:float64:v1')
    elif task != 'binary':
        raise ValueError('Unsupported fingerprint task')
    for X, y in splits:
        digest.update(struct.pack("!QQ", len(X), len(X[0])))
        for row, label in zip(X, y):
            for value in row:
                digest.update(b"missing!" if math.isnan(value) else struct.pack("!d", value))
            digest.update(struct.pack('!d', label) if task == 'regression' else bytes([label]))
    return digest.hexdigest()


class MorphTreeClassifier:
    """Binary dense-numeric classifier. The learner is always MorphTree itself."""
    def __init__(self, *, config=None, genome=None, params=None, backend='pure', tuner=None):
        self.config = MorphTreeConfig() if config is None else config
        if tuner is not None:
            self.config = replace(self.config, tuner=tuner)
        if backend not in ('pure', 'numpy'):
            raise ValueError("backend must be pure or numpy")
        self.backend = backend
        self.base_genome = AlgorithmGenome() if genome is None else genome
        self.base_params = HyperParams() if params is None else params
        if not isinstance(self.config, MorphTreeConfig) or not isinstance(self.base_genome, AlgorithmGenome) or not isinstance(self.base_params, HyperParams):
            raise ValueError("config, genome and params must use their corresponding MorphTree types")
        self._validate_backend_config()
        self.model_ = None
        self.genome_, self.params_, self.report_ = None, None, None
        self.n_features_in_, self.feature_names_in_ = None, None
        self._algorithm_locked = False
        self._training_partition = None
        self.optuna_study_ = None
        self.classes_ = (0, 1)

    def _validate_backend_config(self):
        pass

    def _make_backend(self):
        if self.backend == 'numpy':
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
        return PythonBackend()

    def _make_tuner(self):
        if self.config.tuner == 'optuna':
            from .optuna_tuner import OptunaTuner
            return OptunaTuner()
        from .tuning import MorphTuner
        return MorphTuner()

    def fit(self, X, y, *, search=None, guard=None, feature_names=None, cancel=None):
        start = time.perf_counter()
        control = RunControl(cancel, self.config.search_timeout)
        tuning_enabled = self.config.enable_tuning and self.config.tuner != 'none'
        inferred = get_feature_names(X)
        if feature_names is not None:
            validate_names(feature_names, len(feature_names))
            if inferred is not None and list(feature_names) != inferred:
                raise ValueError("Explicit feature_names must match training DataFrame order")
        names = inferred if feature_names is None else list(feature_names)
        X, y = validate_xy(X, y)
        width = len(X[0])
        validate_names(names, width)
        if width > LoadLimits().max_features:
            raise ValueError("Feature count exceeds the supported model-format limit")
        if (search is None) != (guard is None):
            raise ValueError("Provide both SEARCH and GUARD, or neither")
        if search is None and self._training_partition == 'explicit':
            raise ValueError('This learning recipe requires explicit SEARCH and GUARD splits')
        if search is None and (self._training_partition == 'automatic' or self.config.enable_morphing or tuning_enabled):
            train_ids, search_ids, guard_ids = split_indices(y, self.config.search_fraction, self.config.guard_fraction, self.config.seed)
            train, search_data, guard_data = [subset(X, y, ids) for ids in (train_ids, search_ids, guard_ids)]
            split_kind = "automatic_stratified_disjoint"
            full_X, full_y = X, y
        elif search is not None:
            if not isinstance(search, (tuple, list)) or len(search) != 2 or not isinstance(guard, (tuple, list)) or len(guard) != 2:
                raise ValueError("SEARCH and GUARD must be (X, y) pairs")
            train = (X, y)
            search_data = validate_xy(*search, width, names)
            guard_data = validate_xy(*guard, width, names)
            full_X, full_y = X + search_data[0] + guard_data[0], y + search_data[1] + guard_data[1]
            split_kind = "explicit_caller_responsible_for_disjointness"
        else:
            # Pure baseline needs no automatic data split. These losses are training diagnostics only.
            train = search_data = guard_data = (X, y)
            full_X, full_y = X, y
            split_kind = "baseline_no_selection_metrics_are_training_only"
        X_train, y_train = train
        X_search, y_search = search_data
        X_guard, y_guard = guard_data
        backend = self._make_backend()
        cache = TrainingCache(X_train, X_search, backend)

        def evaluate(genome, params, *, callback=None, early_stopping_rounds=0):
            return BinaryHistogramBooster(genome, params, self.config.seed, backend, memory_enabled=self.config.memory_enabled,
                                          collect_diagnostics=self.config.enable_morphing and not self._algorithm_locked).fit(
                X_train, y_train, eval_set=search_data, callback=callback, cache=cache,
                early_stopping_rounds=early_stopping_rounds)

        base_start = time.perf_counter()
        def base_callback(iteration, loss):
            if cancel is not None and cancel():
                raise SearchInterrupted('cancelled')
            return True
        try:
            incumbent = evaluate(self.base_genome, self.base_params,
                                 callback=base_callback if not self.config.enable_morphing and not tuning_enabled else None)
        except SearchInterrupted as exc:
            raise RuntimeError('Base training cancelled; fitted state was not changed') from exc
        base_time = time.perf_counter() - base_start
        base_search = logloss(y_search, incumbent.predict_proba(X_search))
        base_guard = logloss(y_guard, incumbent.predict_proba(X_guard))
        discovery_start = time.perf_counter()
        if self.config.enable_morphing and not self._algorithm_locked:
            from .mutations import StructuralSearch
            searcher = StructuralSearch(self.config)
            searcher.control = control
            incumbent, structural = searcher.run(
                incumbent, evaluate, X_train, y_train, X_search, y_search, X_guard, y_guard)
        else:
            structural = {"stop_reason": "frozen_species_reused" if self._algorithm_locked else "morphing_disabled",
                          "plateau_reached": False, "history": [], "profiles": [], "coverage": {},
                          "scope": "not searched"}
        discovery_time = time.perf_counter() - discovery_start
        structure_search = logloss(y_search, incumbent.predict_proba(X_search))
        structure_guard = logloss(y_guard, incumbent.predict_proba(X_guard))
        frozen = incumbent.genome.fingerprint()
        tuning_start = time.perf_counter()
        budget_freeze = self.config.freeze_on_budget and structural['stop_reason'] == 'budget_exhausted'
        fixed_base = not self.config.enable_morphing
        can_tune = structural["plateau_reached"] or self._algorithm_locked or budget_freeze or fixed_base
        freeze_reason = ('budget_limited_freeze' if budget_freeze else 'explicit_freeze' if self._algorithm_locked
                         else 'fixed_configuration' if fixed_base else structural['stop_reason'])
        tuning = {"attempted": False, "completed": False, "accepted": False, "trials": [],
                  "reason": "disabled" if not tuning_enabled else "no_verified_plateau", "frozen_fingerprint": frozen}
        tuner = None
        if tuning_enabled and can_tune:
            tuner = self._make_tuner()
            tuner.cancel = cancel
            if hasattr(tuner, "context"):
                tuner.context = {'data': dataset_fingerprint((train, search_data, guard_data)), 'schema': names,
                                 'backend': self.backend, 'memory_enabled': self.config.memory_enabled}
            incumbent, tuning = tuner.optimize(incumbent, evaluate, X_search, y_search, X_guard, y_guard, self.config, base_guard)
        tuning_time = time.perf_counter() - tuning_start
        if incumbent.genome.fingerprint() != frozen:
            raise RuntimeError("Structural fingerprint changed after freeze")
        final_search = logloss(y_search, incumbent.predict_proba(X_search))
        final_guard = logloss(y_guard, incumbent.predict_proba(X_guard))
        params, genome = incumbent.params, incumbent.genome
        refit_start = time.perf_counter()
        final_model = incumbent
        cancelled = structural['stop_reason'] == 'cancelled' or tuning.get('stop_reason') == 'cancelled'
        did_refit = self.config.refit_full and not cancelled and split_kind != "baseline_no_selection_metrics_are_training_only"
        if did_refit:
            refit_control = RunControl(cancel)
            def refit_callback(iteration, loss):
                refit_control.check()
                return True
            refit_control.check()
            final_model = BinaryHistogramBooster(genome, params, self.config.seed, backend, memory_enabled=self.config.memory_enabled).fit(
                full_X, full_y, callback=refit_callback)
        refit_time = time.perf_counter() - refit_start
        report = {
            "version": VERSION, "spec_version": SPEC_VERSION, "python": platform.python_version(), "backend": backend.name,
            "config": public_config(self.config), "stop_reason": structural["stop_reason"], "plateau_reached": structural["plateau_reached"],
            "freeze_reason": freeze_reason,
            "plateau_scope": structural["scope"], "coverage": structural["coverage"], "base_genome": self.base_genome.to_dict(),
            "final_genome": genome.to_dict(), "final_fingerprint": frozen, "base_parameters": asdict(self.base_params),
            "final_parameters": asdict(params), "active_parameters": list(genome.active_parameters()),
            "base_search_loss": base_search, "base_guard_loss": base_guard, "structure_search_loss": structure_search,
            "structure_guard_loss": structure_guard, "final_search_loss": final_search, "final_guard_loss": final_guard,
            "structural_gain": base_search - structure_search, "tuning_gain": structure_search - final_search,
            "morph_history": structural["history"], "error_profiles": structural["profiles"], "tuning": tuning,
            "split": {"kind": split_kind, "train": len(X_train), "search": len(X_search), "guard": len(X_guard),
                      "sealed": "not accepted by fit; evaluate externally", "refit_rows": len(full_X) if did_refit else len(X_train)},
            "feature_schema": {"count": width, "names": names}, "refit_full": did_refit,
            "structural_sampling": {"row_fraction": self.base_params.row_fraction, "train_rows": len(y_train),
                                    "sampled_rows": self.base_params.row_sample_count(len(y_train)),
                                    "policy_active": self.base_params.row_sample_count(len(y_train)) < len(y_train),
                                    "scope": "fixed throughout discovery; numerical tuning only after freeze"},
            "metrics_scope": ("TRAIN diagnostics only; no validation performed" if split_kind == "baseline_no_selection_metrics_are_training_only"
                              else "before refit; SEARCH/GUARD are selection sets, not independent test results"),
            "timings_seconds": {"base": base_time, "discovery": discovery_time, "tuning": tuning_time,
                                "refit": refit_time, "total_fit": time.perf_counter() - start},
        }
        # Commit fitted state only after the entire operation succeeds.
        recipe = {name: getattr(self.config, name) for name in RECIPE_FIELDS}
        recipe['refit_full'] = did_refit
        recipe['partition'] = 'automatic' if split_kind.startswith('automatic') else ('explicit' if search is not None else 'all')
        algorithm = AlgorithmSpecies(genome, params, {'learning_recipe': recipe})
        self.model_, self.genome_, self.params_, self.report_ = final_model, genome, params, FitReport(report)
        self.n_features_in_, self.feature_names_in_ = width, names
        self.optuna_study_ = getattr(tuner, "study", None)
        self.algorithm_ = algorithm
        self.morph_history_ = copy.deepcopy(structural["history"])
        return self

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("MorphTreeClassifier is not fitted")

    def predict_proba(self, X):
        self._require_fitted()
        rows = validate_X(X, self.n_features_in_, self.feature_names_in_, allow_empty=True)
        probabilities = self.model_.predict_proba(rows) if rows else []
        return [[1 - p, p] for p in probabilities]

    def predict(self, X):
        return [int(row[1] >= 0.5) for row in self.predict_proba(X)]

    def score(self, X, y):
        probabilities = [row[1] for row in self.predict_proba(X)]
        return accuracy(validate_y(y, len(probabilities)), probabilities)

    def report(self):
        self._require_fitted()
        return self.report_.to_dict()

    def inspect(self):
        return self.report()

    def inspect_algorithm(self):
        self._require_fitted()
        return self.algorithm_.to_dict()

    def save_algorithm(self, path):
        self._require_fitted()
        self.algorithm_.save(path)

    def save_model(self, path):
        self._require_fitted()
        write_json(path, {"format": "MorphTreeModel", "spec_version": SPEC_VERSION, "producer_version": VERSION,
                          "algorithm": self.algorithm_.to_dict(), "n_features": self.n_features_in_,
                          "feature_names": self.feature_names_in_, "booster": self.model_.to_dict(), "report": self.report()})

    save = save_model

    @classmethod
    def load_model(cls, path, *, limits=None, backend='pure'):
        return cls._from_payload(read_json(path, limits), limits=limits, backend=backend)

    @classmethod
    def _from_payload(cls, data, *, limits=None, backend='pure'):
        limits = limits or LoadLimits()
        require_keys(data, ("format", "spec_version", "producer_version", "algorithm", "n_features", "feature_names", "booster", "report"))
        if data["format"] != "MorphTreeModel" or type(data["spec_version"]) is not int or data["spec_version"] != SPEC_VERSION:
            raise ValueError("Unsupported model format/version")
        genome, params, metadata = decode_species(data["algorithm"])
        recipe = metadata.get('learning_recipe')
        settings = recipe_settings(recipe)
        if recipe is None:
            settings = replace(settings, memory_enabled=True)  # Legacy artifacts predate explicit recipes.
        obj = cls(config=replace(settings, enable_morphing=False, enable_tuning=False), genome=genome, params=params, backend=backend)
        count = integer(data["n_features"], "n_features", 1, limits.max_features)
        validate_names(data["feature_names"], count)
        obj.model_ = decode_booster(data["booster"], genome, params, obj._make_backend(), limits)
        if obj.model_.n_features_ != count:
            raise ValueError("Model and booster dimensions disagree")
        if not isinstance(data["report"], dict):
            raise ValueError("Model report must be an object")
        obj.n_features_in_, obj.feature_names_in_ = count, data["feature_names"]
        obj.genome_, obj.params_, obj.report_ = genome, params, FitReport(imported_report(data["report"]))
        obj.algorithm_ = AlgorithmSpecies(genome, params, metadata)
        obj.morph_history_ = []
        obj._algorithm_locked = True
        obj._training_partition = recipe['partition'] if recipe else None
        return obj

    load = load_model

    @classmethod
    def from_algorithm(cls, algorithm, *, retune=False, config=None, backend='pure', tuner=None):
        species = algorithm if isinstance(algorithm, AlgorithmSpecies) else AlgorithmSpecies.load(algorithm)
        if type(retune) is not bool:
            raise ValueError("retune must be a bool")
        recipe = species.metadata.get('learning_recipe')
        settings = replace(recipe_settings(recipe, config), enable_morphing=False, enable_tuning=retune)
        if recipe is None and config is None:
            settings = replace(settings, memory_enabled=True)
        if retune and (tuner or settings.tuner) == 'none':
            raise ValueError("retune=True requires an explicit native or optuna tuner")
        obj = cls(config=settings, genome=species.genome, params=species.parameters, backend=backend, tuner=tuner)
        obj._algorithm_locked = True
        obj._training_partition = recipe['partition'] if recipe else None
        return obj

    @classmethod
    def discover(cls, X, y, *, config=None, params=None, genome=None, search=None, guard=None, backend='pure', tuner=None):
        """Explicit experimental discovery; no generalization advantage is promised."""
        settings = replace(config or MorphTreeConfig(), enable_morphing=True, refit_full=False)
        model = cls(config=settings, params=params, genome=genome, backend=backend, tuner=tuner).fit(X, y, search=search, guard=guard)
        return AlgorithmSpecies(model.genome_, model.params_, {**model.algorithm_.metadata, "discovery_report": model.report()})

    def export_algorithm(self, path):
        self._require_fitted()
        self.algorithm_.export_python(path)

    def export_python(self, path):
        self._require_fitted()
        from .exporting import export_predictor
        export_predictor(self, path)

    def export_bundle(self, path):
        self._require_fitted()
        from .exporting import export_bundle
        export_bundle(self, path)

    def save_report(self, path):
        from .exporting import report_markdown
        from .persistence import atomic_text
        if str(path).lower().endswith(".md"):
            atomic_text(path, report_markdown(self.report()))
        else:
            write_json(path, self.report())


class MorphTreeRegressor:
    """Fixed native Base with squared error; no classification/morphing wrapper.

    This first development increment has no numeric tuner or structural discovery.
    Validation or explicit Optuna TRAIN/SEARCH/GUARD selects numeric values only.
    """
    def __init__(self, *, config=None, params=None, backend='pure', tuner=None, genome=None):
        config = MorphTreeConfig() if config is None else config
        params = HyperParams() if params is None else params
        genome = AlgorithmGenome() if genome is None else genome
        if not isinstance(config, MorphTreeConfig) or not isinstance(params, HyperParams) or not isinstance(genome, AlgorithmGenome):
            raise ValueError('config, params and genome must use their MorphTree types')
        if tuner is not None:
            config = replace(config, tuner=tuner)
        if backend not in ('pure', 'numpy'):
            raise ValueError('backend must be pure or numpy')
        validate_task_genome('regression',genome)
        if config.enable_morphing or config.memory_enabled or config.diagnostic_guidance:
            raise ValueError('Regression currently supports fixed Base without morphing, memory or guidance')
        if config.tuner not in ('none', 'optuna'):
            raise ValueError('Regression supports tuner=none or explicit tuner=optuna')
        if config.search_timeout is not None:
            raise ValueError('Regression search_timeout is not supported; no structural search runs')
        self.config, self.base_params, self.base_genome = config, params, genome
        self.backend, self.task = backend, 'regression'
        self.model_, self.params_, self.genome_, self.report_ = None, None, None, None
        self.n_features_in_, self.feature_names_in_ = None, None
        self.optuna_study_ = None

    def _make_backend(self):
        if self.backend == 'numpy':
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
        return PythonBackend()

    def fit(self, X, y, *, validation=None, search=None, guard=None, feature_names=None, cancel=None):
        start = time.perf_counter()
        control = RunControl(cancel)
        inferred = get_feature_names(X)
        if feature_names is not None:
            validate_names(feature_names, len(feature_names))
            if inferred is not None and list(feature_names) != inferred:
                raise ValueError('Explicit feature_names must match training DataFrame order')
        names = inferred if feature_names is None else list(feature_names)
        rows = validate_X(X)
        targets = validate_regression_y(y, len(rows))
        width = len(rows[0])
        validate_names(names, width)
        limits = LoadLimits()
        if width > limits.max_features:
            raise ValueError('Feature count exceeds the supported model-format limit')
        tuning_enabled = self.config.enable_tuning and self.config.tuner == 'optuna'
        if (search is None) != (guard is None):
            raise ValueError('Provide both SEARCH and GUARD, or neither')
        if validation is not None and (tuning_enabled or search is not None):
            raise ValueError('validation is exclusive with tuning/SEARCH/GUARD; never supply final test data')
        def selection_pair(pair, label):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(label + ' must be an (X, y) pair; never supply final test data')
            values = validate_X(pair[0], width, names)
            return values, validate_regression_y(pair[1], len(values))

        full_rows, full_targets = rows, targets
        search_data = guard_data = None
        if search is not None:
            search_data, guard_data = selection_pair(search, 'SEARCH'), selection_pair(guard, 'GUARD')
            full_rows = rows + search_data[0] + guard_data[0]
            full_targets = targets + search_data[1] + guard_data[1]
            split_kind = 'explicit_caller_responsible_for_disjointness'
        elif tuning_enabled:
            partitions = regression_split_indices(len(rows), self.config.search_fraction, self.config.guard_fraction, self.config.seed)
            (rows, targets), search_data, guard_data = [subset(rows, targets, ids) for ids in partitions]
            split_kind = 'automatic_random_disjoint'
        selected_validation = search_data
        if validation is not None:
            selected_validation = selection_pair(validation, 'validation')
            full_rows, full_targets = rows + selected_validation[0], targets + selected_validation[1]
        backend = self._make_backend()
        cache = TrainingCache(rows, selected_validation[0], backend) if selected_validation else None
        def callback(iteration, loss):
            control.check()
            return True
        def evaluate(genome, params, *, callback=callback, early_stopping_rounds=0):
            return BinaryHistogramBooster(genome, params, self.config.seed, backend,
                memory_enabled=False, objective='squared_error').fit(rows, targets, eval_set=selected_validation,
                early_stopping_rounds=early_stopping_rounds, callback=callback, cache=cache)
        tuning, study = {'attempted': False, 'reason': 'disabled'}, None
        try:
            control.check()
            # GUARD compares against an immutable full-budget TRAIN Base, never a SEARCH-selected prefix.
            base = evaluate(self.base_genome, self.base_params,
                early_stopping_rounds=self.config.early_stopping_rounds if validation is not None else 0)
            selected = base
            if tuning_enabled:
                from .optuna_tuner import OptunaTuner
                from .metrics import rmse
                tuner = OptunaTuner()
                tuner.cancel = cancel
                tuner.context = {'data': dataset_fingerprint(((rows, targets), search_data, guard_data), task='regression'),
                    'schema': names, 'backend': self.backend, 'task': 'regression', 'objective': 'squared_error',
                    'partition': split_kind, 'memory_enabled': False}
                selected, tuning = tuner.optimize(base, evaluate, *search_data, *guard_data, self.config,
                                                   rmse(guard_data[1], base.predict_raw(guard_data[0])))
                study = tuner.study
            params = replace(selected.params, n_estimators=len(selected.trees))
            train_metrics = regression_metrics(targets, selected.predict_raw(rows))
            validation_metrics = (regression_metrics(selected_validation[1], selected.predict_raw(selected_validation[0]))
                                  if selected_validation else None)
            guard_metrics = regression_metrics(guard_data[1], selected.predict_raw(guard_data[0])) if guard_data else None
            final = selected
            interrupted = tuning.get('stop_reason') in ('cancelled', 'timeout')
            did_refit = bool(selected_validation and self.config.refit_full and not interrupted)
            if did_refit:
                control.check()
                final = BinaryHistogramBooster(self.base_genome, params, self.config.seed, backend,
                    memory_enabled=False, objective='squared_error').fit(
                    full_rows, full_targets, callback=callback)
            final.params = params
            # Check representation budgets without allocating a second decoded ensemble.
            if sum(len(t.nodes) for t in final.trees) > limits.max_nodes or sum(map(len, final.binner.thresholds)) > limits.max_thresholds:
                raise ValueError('Fitted regression model exceeds default format budgets')
            if abs(final.base_score) > 1e100 or any(not math.isfinite(node.value) or abs(node.value) > 1e100
                for tree in final.trees for node in tree.nodes):
                raise ArithmeticError('Regression state exceeds the supported finite model range')
            report = {'version': VERSION, 'spec_version': SPEC_VERSION, 'task': 'regression', 'objective': 'squared_error',
                'backend': backend.name, 'config': public_config(self.config), 'freeze_reason': 'fixed_configuration',
                'stop_reason': tuning.get('stop_reason', 'numeric_tuning_complete' if tuning_enabled else
                    'early_stopping' if selected.stopped_early_ else 'fixed_base_complete'),
                'base_parameters': asdict(self.base_params), 'final_parameters': asdict(params),
                'selected_trees': len(selected.trees), 'validation_rmse_curve': list(selected.search_curve_),
                'train_metrics': train_metrics, 'validation_metrics': validation_metrics, 'refit_full': did_refit,
                'split': {'kind': 'explicit_validation' if selected_validation else 'all_train', 'train': len(rows),
                    'validation': len(selected_validation[1]) if selected_validation else 0,
                    'refit_rows': len(rows) + len(selected_validation[1]) if did_refit else len(rows)},
                'metrics_scope': 'TRAIN/validation before optional refit; selection metrics, not an independent test'
                    if selected_validation else 'TRAIN diagnostics only; no validation performed',
                'timings_seconds': {'total_fit': time.perf_counter() - start}}
            if search_data:
                report.update(tuning=tuning, search_metrics=validation_metrics, guard_metrics=guard_metrics,
                    metric='rmse', frozen_fingerprint=self.base_genome.fingerprint(),
                    base_search_rmse=regression_metrics(search_data[1], base.predict_raw(search_data[0]))['rmse'],
                    base_guard_rmse=regression_metrics(guard_data[1], base.predict_raw(guard_data[0]))['rmse'],
                    metrics_scope='TRAIN/SEARCH/GUARD before optional refit; selection diagnostics, not independent test',
                    split={'kind': split_kind, 'train': len(targets), 'search': len(search_data[1]),
                        'guard': len(guard_data[1]), 'refit_rows': len(full_targets) if did_refit else len(targets)})
        except SearchInterrupted as exc:
            raise RuntimeError('Regression fit cancelled; fitted state was not changed') from exc
        # No partially fitted state escapes validation, cancellation, refit or metrics failures.
        self.model_, self.params_, self.genome_, self.report_ = final, params, self.base_genome, FitReport(report)
        self.n_features_in_, self.feature_names_in_ = width, names
        self.optuna_study_ = study
        return self

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError('MorphTreeRegressor is not fitted')

    def predict(self, X):
        self._require_fitted()
        rows = validate_X(X, self.n_features_in_, self.feature_names_in_, allow_empty=True)
        prediction = [float(v) for v in self.model_.predict_raw(rows)] if rows else []
        if any(not math.isfinite(v) for v in prediction):
            raise ArithmeticError('Regression prediction outside finite floating-point range')
        return prediction

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    def report(self):
        self._require_fitted()
        return self.report_.to_dict()

    def inspect(self):
        return self.report()

    def inspect_algorithm(self):
        self._require_fitted()
        return {'task': 'regression', 'objective': 'squared_error', 'genome': self.genome_.to_dict(),
                'parameters': asdict(self.params_)}

    def _model_payload(self, *, include_report=True):
        self._require_fitted()
        return {**({'genome':self.genome_.to_dict()} if self.genome_!=AlgorithmGenome() else {}), 'format': TASK_GROWTH_FORMATS['regression'] if self.genome_!=AlgorithmGenome() else 'MorphTreeRegressionModel', 'spec_version': SPEC_VERSION, 'producer_version': VERSION,
                'task': 'regression', 'objective': 'squared_error', 'parameters': asdict(self.params_),
                'n_features': self.n_features_in_, 'feature_names': self.feature_names_in_,
                'booster': self.model_.to_dict(), 'report': self.report() if include_report else {}}

    def save_model(self, path):
        write_json(path, self._model_payload())

    save = save_model

    @classmethod
    def load_model(cls, path, *, limits=None, backend='pure'):
        return cls._from_payload(read_json(path, limits), limits=limits, backend=backend)

    @classmethod
    def _from_payload(cls, data, *, limits=None, backend='pure'):
        limits = limits or LoadLimits()
        genome=decode_task_genome('regression',data)
        require_keys({k:v for k,v in data.items() if k!='genome'}, ('format', 'spec_version', 'producer_version', 'task', 'objective', 'parameters',
                           'n_features', 'feature_names', 'booster', 'report'))
        if (data['format'] not in ('MorphTreeRegressionModel',TASK_GROWTH_FORMATS['regression']) or type(data['spec_version']) is not int
            or data['spec_version'] != SPEC_VERSION or data['task'] != 'regression' or data['objective'] != 'squared_error'):
            raise ValueError('Unsupported regression model format/task/objective/version')
        require_keys(data['parameters'], [field.name for field in fields(HyperParams)])
        params = HyperParams(**data['parameters'])
        obj = cls(params=params, backend=backend, genome=genome)
        count = integer(data['n_features'], 'n_features', 1, limits.max_features)
        validate_names(data['feature_names'], count)
        obj.model_ = decode_booster(data['booster'], genome, params, obj._make_backend(), limits,
                                    objective='squared_error')
        if obj.model_.n_features_ != count or not isinstance(data['report'], dict):
            raise ValueError('Invalid regression dimensions/report')
        obj.params_, obj.genome_, obj.report_ = params, genome, FitReport(imported_regression_report(data['report']))
        obj.n_features_in_ = count
        obj.feature_names_in_ = list(data['feature_names']) if data['feature_names'] is not None else None
        return obj

    load = load_model

    def export_python(self, path):
        self._require_fitted()
        from .exporting import export_predictor
        export_predictor(self, path)

    def export_bundle(self, path):
        self._require_fitted()
        from .exporting import export_bundle
        export_bundle(self, path)

    def save_report(self, path):
        from .persistence import atomic_text
        report = self.report()
        if str(path).lower().endswith('.md'):
            text = '# MorphTree regression report\n\nSelection diagnostics; never independent test evidence.\n\n'
            text += '```json\n' + json.dumps(report, indent=2, allow_nan=False) + '\n```\n'
            atomic_text(path, text)
        else:
            write_json(path, report)


class MorphTreeMulticlassClassifier:
    """Fixed native multiclass Base. Explicit softmax or normalized OvR, no search."""
    def __init__(self, *, config=None, params=None, backend='pure', strategy='ovr', genome=None):
        config = MorphTreeConfig() if config is None else config
        params = HyperParams() if params is None else params
        genome=validate_task_genome('multiclass',AlgorithmGenome() if genome is None else genome)
        self.base_genome=genome;self.genome_=None
        if not isinstance(config, MorphTreeConfig) or not isinstance(params, HyperParams):
            raise ValueError('config and params must use their MorphTree types')
        if backend not in ('pure', 'numpy') or strategy not in ('softmax', 'ovr'):
            raise ValueError('backend must be pure/numpy and strategy softmax/ovr')
        if (config.enable_morphing or config.memory_enabled or config.diagnostic_guidance or config.tuner != 'none'
            or config.search_timeout is not None or config.tune_timeout is not None):
            raise ValueError('Multiclass currently supports fixed Base without tuning, morphing, memory or guidance')
        self.config, self.base_params, self.backend, self.strategy = config, params, backend, strategy
        self.task = 'multiclass'
        self.model_, self.params_, self.report_, self.classes_ = None, None, None, None
        self.n_features_in_, self.feature_names_in_ = None, None

    def _make_backend(self):
        if self.backend == 'numpy':
            from .numpy_backend import NumpyBackend
            return NumpyBackend()
        return PythonBackend()

    def fit(self, X, y, *, validation=None, feature_names=None, cancel=None):
        start, control = time.perf_counter(), RunControl(cancel)
        names = get_feature_names(X)
        if feature_names is not None:
            validate_names(feature_names, len(feature_names))
            if names is not None and list(feature_names) != names:
                raise ValueError('Explicit feature_names must match training DataFrame order')
            names = list(feature_names)
        rows = validate_X(X)
        targets, classes = validate_multiclass_y(y, len(rows))
        width, k, limits = len(rows[0]), len(classes), LoadLimits()
        validate_names(names, width)
        if width > limits.max_features or k*self.base_params.n_estimators > limits.max_trees:
            raise ValueError('Multiclass feature/aggregate tree budget exceeded; reduce rounds or classes')
        selected_validation = None
        if validation is not None:
            if not isinstance(validation, (tuple, list)) or len(validation) != 2:
                raise ValueError('validation must be an (X, y) selection pair; never pass final test data')
            vx = validate_X(validation[0], width, names)
            vy, _ = validate_multiclass_y(validation[1], len(vx), classes)
            selected_validation = vx, vy
        backend = self._make_backend()
        def callback(iteration, loss):
            control.check()
            return True
        def make(params):
            return BinaryHistogramBooster(self.base_genome, params, self.config.seed, backend,
                memory_enabled=False, objective=self.strategy, n_classes=k)
        try:
            control.check()
            selected = make(self.base_params).fit(rows, targets, eval_set=selected_validation,
                early_stopping_rounds=self.config.early_stopping_rounds, callback=callback)
            rounds = len(selected.trees)//k
            params = replace(self.base_params, n_estimators=rounds)
            train_metrics = multiclass_metrics(targets, selected.predict_proba(rows))
            validation_metrics = (multiclass_metrics(selected_validation[1], selected.predict_proba(selected_validation[0]))
                                  if selected_validation else None)
            did_refit = bool(selected_validation and self.config.refit_full)
            final = selected
            if did_refit:
                final = make(params).fit(rows+selected_validation[0], targets+selected_validation[1], callback=callback)
            final.params = params
            if (sum(len(tree.nodes) for tree in final.trees) > limits.max_nodes
                or sum(map(len, final.binner.thresholds)) > limits.max_thresholds):
                raise ValueError('Fitted multiclass model exceeds default format budgets')
            if any(not math.isfinite(node.value) or abs(node.value)>1e100 for tree in final.trees for node in tree.nodes):
                raise ArithmeticError('Multiclass state exceeds finite format range')
            report = {'version': VERSION, 'spec_version': SPEC_VERSION, 'task': 'multiclass', 'strategy': self.strategy,
                'backend': backend.name, 'classes': list(classes), 'config': public_config(self.config),
                'freeze_reason': 'fixed_configuration', 'stop_reason': 'early_stopping' if selected.stopped_early_ else 'fixed_base_complete',
                'base_parameters': asdict(self.base_params), 'final_parameters': asdict(params), 'selected_rounds': rounds,
                'tree_count': len(final.trees), 'validation_logloss_curve': list(selected.search_curve_),
                'train_metrics': train_metrics, 'validation_metrics': validation_metrics, 'refit_full': did_refit,
                'hessian': 'diagonal_upper_bound_2p(1-p)' if self.strategy=='softmax' else 'independent_binary_p(1-p)',
                'split': {'kind': 'explicit_validation' if selected_validation else 'all_train', 'train': len(rows),
                    'validation': len(selected_validation[1]) if selected_validation else 0,
                    'refit_rows': len(rows)+len(selected_validation[1]) if did_refit else len(rows)},
                'metrics_scope': 'TRAIN/validation before optional refit; selection diagnostics, not independent test',
                'timings_seconds': {'total_fit': time.perf_counter()-start}}
        except SearchInterrupted as exc:
            raise RuntimeError('Multiclass fit cancelled; fitted state was not changed') from exc
        self.model_, self.params_, self.report_, self.classes_ = final, params, FitReport(report), classes
        self.genome_=self.base_genome
        self.n_features_in_, self.feature_names_in_ = width, names
        return self

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError('MorphTreeMulticlassClassifier is not fitted')

    def predict_proba(self, X):
        self._require_fitted()
        rows = validate_X(X, self.n_features_in_, self.feature_names_in_, allow_empty=True)
        return self.model_.predict_proba(rows) if rows else []

    def predict(self, X):
        return [self.classes_[max(range(len(row)), key=row.__getitem__)] for row in self.predict_proba(X)]

    def score(self, X, y):
        probabilities = self.predict_proba(X)
        labels, _ = validate_multiclass_y(y, len(probabilities), self.classes_)
        return multiclass_metrics(labels, probabilities)['accuracy']

    def report(self):
        self._require_fitted()
        return self.report_.to_dict()

    def inspect(self):
        return self.report()

    def inspect_algorithm(self):
        self._require_fitted()
        return {'task': self.task, 'strategy': self.strategy, 'classes': list(self.classes_),
                'genome': self.genome_.to_dict(), 'parameters': asdict(self.params_)}

    def _model_payload(self, *, include_report=True):
        self._require_fitted()
        return {**({'genome':self.genome_.to_dict()} if self.genome_!=AlgorithmGenome() else {}), 'format': TASK_GROWTH_FORMATS['multiclass'] if self.genome_!=AlgorithmGenome() else 'MorphTreeMulticlassModel', 'spec_version': SPEC_VERSION, 'producer_version': VERSION,
            'task': 'multiclass', 'strategy': self.strategy, 'classes': list(self.classes_), 'parameters': asdict(self.params_),
            'n_features': self.n_features_in_, 'feature_names': self.feature_names_in_, 'booster': self.model_.to_dict(),
            'report': self.report() if include_report else {}}

    def save_model(self, path):
        write_json(path, self._model_payload())

    save = save_model

    @classmethod
    def load_model(cls, path, *, limits=None, backend='pure'):
        return cls._from_payload(read_json(path, limits), limits=limits, backend=backend)

    @classmethod
    def _from_payload(cls, data, *, limits=None, backend='pure'):
        limits = limits or LoadLimits()
        genome=decode_task_genome('multiclass',data)
        require_keys({k:v for k,v in data.items() if k!='genome'}, ('format', 'spec_version', 'producer_version', 'task', 'strategy', 'classes', 'parameters',
                           'n_features', 'feature_names', 'booster', 'report'))
        if (data['format'] not in ('MorphTreeMulticlassModel',TASK_GROWTH_FORMATS['multiclass']) or type(data['spec_version']) is not int
            or data['spec_version'] != SPEC_VERSION or data['task'] != 'multiclass'):
            raise ValueError('Unsupported multiclass format/version/task')
        vocabulary = data['classes']
        if not isinstance(vocabulary, list):
            raise ValueError('classes must be a sorted list')
        _, classes = validate_multiclass_y(vocabulary, len(vocabulary))
        if list(classes) != vocabulary:
            raise ValueError('classes must be sorted and unique')
        require_keys(data['parameters'], [field.name for field in fields(HyperParams)])
        params = HyperParams(**data['parameters'])
        obj = cls(params=params, backend=backend, strategy=data['strategy'], genome=genome)
        count = integer(data['n_features'], 'n_features', 1, limits.max_features)
        validate_names(data['feature_names'], count)
        booster = decode_multiclass_booster(data['booster'], params, obj._make_backend(), len(classes), obj.strategy, limits, genome=genome)
        if booster.n_features_ != count or not isinstance(data['report'], dict):
            raise ValueError('Invalid multiclass dimensions/report')
        obj.model_, obj.params_, obj.classes_ = booster, params, classes
        obj.genome_=genome
        obj.report_ = FitReport(imported_multiclass_report(data['report']))
        obj.n_features_in_ = count
        obj.feature_names_in_ = list(data['feature_names']) if data['feature_names'] is not None else None
        return obj

    load = load_model

    def export_python(self, path):
        self._require_fitted()
        from .exporting import export_predictor
        export_predictor(self, path)

    def export_bundle(self, path):
        self._require_fitted()
        from .exporting import export_bundle
        export_bundle(self, path)

    def save_report(self, path):
        from .persistence import atomic_text
        if str(path).lower().endswith('.md'):
            atomic_text(path, '# MorphTree multiclass report\n\nSelection diagnostics, not independent test evidence.\n\n```json\n'
                        +json.dumps(self.report(), indent=2, allow_nan=False)+'\n```\n')
        else:
            write_json(path, self.report())


class AutoMorphTree:
    """Native task dispatch with explicit bounded numeric selection; fixed Base by default."""
    def __init__(self, *, task='auto', mode='fast', config=None, params=None, backend='pure',
                 max_models=4, time_budget=30., candidate_params=None):
        if task not in ('auto', 'binary', 'multiclass', 'regression') or mode not in ('fast', 'numeric'):
            raise ValueError('AutoMorphTree currently supports task auto/binary/multiclass/regression and mode=fast/numeric')
        config = MorphTreeConfig() if config is None else config
        params = HyperParams() if params is None else params
        if not isinstance(config, MorphTreeConfig) or not isinstance(params, HyperParams):
            raise ValueError('config and params must use their MorphTree types')
        if backend not in ('pure', 'numpy'):
            raise ValueError('backend must be pure or numpy')
        if config.tuner != 'none' or config.enable_morphing or config.memory_enabled or config.diagnostic_guidance:
            raise ValueError('AutoMorphTree uses a fixed structure without implicit Optuna, morphing, memory or guidance')
        if type(max_models) is not int or not 1 <= max_models <= 16:
            raise ValueError('max_models must be an integer in [1, 16]')
        if type(time_budget) not in (int, float) or not math.isfinite(time_budget) or time_budget <= 0:
            raise ValueError('time_budget must be positive and finite')
        if candidate_params is not None and (not isinstance(candidate_params, (list, tuple))
                or len(candidate_params) > 15 or any(not isinstance(p, HyperParams) for p in candidate_params)):
            raise ValueError('candidate_params must contain at most 15 HyperParams')
        if mode == 'fast' and (candidate_params is not None or max_models != 4 or time_budget != 30.):
            raise ValueError('Numeric budgets/portfolio require mode=numeric')
        if mode == 'numeric' and (config.search_timeout is not None or config.tune_timeout is not None):
            raise ValueError('Use time_budget for numeric Auto; separate search/tune timeouts are unsupported')
        self.max_models, self.time_budget = max_models, float(time_budget)
        self.candidate_params = None if candidate_params is None else tuple(candidate_params)
        self.selection_report_ = None
        self.task, self.mode, self.config, self.params, self.backend = task, mode, config, params, backend
        self.model_, self.selected_task_ = None, None

    def fit(self, X, y, **kwargs):
        if hasattr(y, 'tolist'):
            y = y.tolist()
        if isinstance(y, (str, bytes)):
            raise ValueError('y must be a one-dimensional target sequence')
        try:
            values = list(y)
        except TypeError as exc:
            raise ValueError('y must be a one-dimensional target sequence') from exc
        task = self.task
        if task == 'auto':
            if values and all(type(v) is str for v in values):
                task = 'multiclass'
            elif values and all(type(v) is int or type(v) is float and math.isfinite(v) for v in values):
                if set(values) == {0, 1}:
                    task = 'binary'
                elif any(type(v) is float and not v.is_integer() for v in values):
                    task = 'regression'
                else:
                    raise ValueError('Integer-valued targets are ambiguous; set task=regression or task=multiclass explicitly')
            else:
                raise ValueError('Cannot infer a task safely; provide valid targets and an explicit task')
        learner = {'binary': MorphTreeClassifier, 'multiclass': MorphTreeMulticlassClassifier, 'regression': MorphTreeRegressor}[task]
        selection = None
        if self.mode == 'numeric':
            candidate, selection = self._fit_numeric(learner, task, X, values, **kwargs)
        else:
            candidate = learner(config=self.config, params=self.params, backend=self.backend).fit(X, values, **kwargs)
        self.model_, self.selected_task_, self.selection_report_ = candidate, task, selection
        return self

    def _fit_numeric(self, learner, task, X, y, *, search=None, guard=None, feature_names=None, cancel=None):
        if search is None or guard is None:
            raise ValueError('Numeric Auto requires explicit disjoint TRAIN, SEARCH and GUARD; never pass final test data')
        names = get_feature_names(X)
        if feature_names is not None:
            validate_names(feature_names, len(feature_names))
            if names is not None and list(feature_names) != names:
                raise ValueError('Explicit feature_names must match training DataFrame order')
            names = list(feature_names)
        rows = validate_X(X)
        validate_names(names, len(rows[0]))
        if task == 'multiclass':
            _, vocabulary = validate_multiclass_y(y, len(rows))
            targets = list(y)
        else:
            vocabulary = None
            targets = validate_regression_y(y, len(rows)) if task == 'regression' else validate_y(y, len(rows), both_classes=True)
        def pair(value):
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError('Selection data must be an (X, y) pair')
            vx = validate_X(value[0], len(rows[0]), names)
            if task == 'multiclass':
                vy, _ = validate_multiclass_y(value[1], len(vx), vocabulary)
            else:
                vy = validate_regression_y(value[1], len(vx)) if task == 'regression' else validate_y(value[1], len(vx), both_classes=True)
            return vx, vy
        sx, sy = pair(search); gx, gy = pair(guard)
        # This finite portfolio is fixed before observing SEARCH/GUARD losses.
        k = len(vocabulary) if vocabulary else 1
        rounds = min(4096 // k, 2 * self.params.n_estimators)
        support = min(self.params.min_samples_leaf, max(2, len(rows) // 40))
        candidates = list(self.candidate_params) if self.candidate_params is not None else [
            replace(self.params, n_estimators=rounds),
            replace(self.params, learning_rate=min(1., 2*self.params.learning_rate),
                    max_leaves=min(4096, 2*self.params.max_leaves), min_samples_leaf=support),
            replace(self.params, l2=min(1e6, max(1., 4*self.params.l2)),
                    min_samples_leaf=max(self.params.min_samples_leaf, max(2, len(rows)//20)))]
        unique = []
        for params in candidates:
            if params != self.params and params not in unique:
                unique.append(params)
        portfolio = [self.params] + unique[:self.max_models-1]
        if any(k*p.n_estimators > 4096 for p in portfolio):
            raise ValueError('Numeric candidate exceeds aggregate tree budget')
        if cancel is not None and not callable(cancel):
            raise ValueError('cancel must be callable or None')
        start = time.monotonic()
        deadline = start + self.time_budget
        cancelled = False
        def interrupt():
            nonlocal cancelled
            if cancel is not None and cancel():
                cancelled = True
                return True
            return time.monotonic() >= deadline
        cfg = replace(self.config, refit_full=False, early_stopping_rounds=0)
        records = []
        best = None
        base_guard = best_guard = best_search = None
        selected_index = 0
        stop = 'portfolio_exhausted'
        attempted = 0
        def loss(model, vx, vy):
            if task == 'regression':
                return regression_metrics(vy, model.predict(vx))['rmse']
            if task == 'binary':
                return logloss(vy, [row[1] for row in model.predict_proba(vx)])
            return multiclass_metrics(vy, model.predict_proba(vx))['logloss']
        for index, params in enumerate(portfolio):
            if interrupt():
                stop = 'cancelled' if cancelled else 'time_budget'
                break
            attempted += 1
            begun = time.monotonic()
            try:
                candidate = learner(config=cfg, params=params, backend=self.backend).fit(rows, targets, feature_names=names, cancel=interrupt)
                search_loss = loss(candidate, sx, sy)
                guard_loss = loss(candidate, gx, gy)
            except RuntimeError as exc:
                if not isinstance(exc.__cause__, SearchInterrupted):
                    raise
                stop = 'cancelled' if cancelled else 'time_budget'
                records.append({'index':index, 'parameters':asdict(params), 'status':'INTERRUPTED', 'fit_seconds':time.monotonic()-begun})
                break
            if index == 0:
                accepted = True
                base_guard = guard_loss
            else:
                accepted = (search_loss < best_search and guard_loss <= best_guard + cfg.guard_tolerance
                            and guard_loss <= base_guard + cfg.global_guard_tolerance)
            records.append({'index':index, 'parameters':asdict(params), 'status':'COMPLETE', 'accepted':accepted,
                            'search_loss':search_loss, 'guard_loss':guard_loss, 'fit_seconds':time.monotonic()-begun})
            if accepted:
                best, best_search, best_guard, selected_index = candidate, search_loss, guard_loss, index
        if cancelled:
            raise RuntimeError('Auto fit cancelled; fitted state was not changed')
        if best is None:
            raise RuntimeError('Time budget could not complete Base; fitted state was not changed')
        # Refit is explicit in config and charged separately. A timeout retains the
        # complete TRAIN winner; cancellation preserves the previous Auto fit.
        refit = False
        refit_status = 'disabled'
        if self.config.refit_full:
            refit_status = 'budget_exhausted'
            if not interrupt():
                attempted += 1
                labels = ([vocabulary[v] for v in sy+gy] if vocabulary else sy+gy)
                try:
                    full = learner(config=cfg, params=best.params_, backend=self.backend).fit(
                        rows+sx+gx, targets+labels, feature_names=names, cancel=interrupt)
                    best, refit, refit_status = full, True, 'complete'
                except RuntimeError as exc:
                    if not isinstance(exc.__cause__, SearchInterrupted):raise
                    refit_status = 'interrupted'
        if cancelled:
            raise RuntimeError('Auto fit cancelled; fitted state was not changed')
        result = {'configuration_selection':'guarded_numeric_portfolio', 'best_configuration_claim':False,
                  'selected_index':selected_index, 'fallback_to_base':selected_index==0,
                  'metric':'rmse' if task=='regression' else 'logloss', 'base_guard_loss':base_guard,
                  'selected_search_loss':best_search, 'selected_guard_loss':best_guard,
                  'attempted_fits':attempted, 'completed_selection_fits':sum(r['status']=='COMPLETE' for r in records),
                  'max_models':self.max_models, 'time_budget_seconds':self.time_budget, 'elapsed_seconds':time.monotonic()-start,
                  'budget_kind':'cooperative at complete boosting rounds; validation/metrics and allocations are not hard bounded',
                  'stop_reason':stop, 'refit_full':refit, 'refit_status':refit_status,
                  'partition':{'kind':'explicit_caller_responsible_for_disjointness','train':len(rows),'search':len(sx),'guard':len(gx)},
                  'candidates':records, 'structure':'immutable fixed Base', 'metrics_scope':'selection diagnostics before optional refit'}
        return best, result

    def report(self):
        if self.model_ is None:
            raise RuntimeError('AutoMorphTree is not fitted')
        result = self.model_.report()
        result['auto'] = {'selected_task': self.selected_task_, 'rule': 'explicit' if self.task != 'auto' else 'conservative_target_type',
                          'configuration_selection': 'fixed_base_no_search', 'best_configuration_claim': False}
        if self.selection_report_ is not None:
            result['auto'].update(copy.deepcopy(self.selection_report_))
        return result

    def save_report(self, path):
        if self.selection_report_ is None:
            if self.model_ is None:
                raise RuntimeError('AutoMorphTree is not fitted')
            return self.model_.save_report(path)
        if str(path).lower().endswith('.md'):
            from .persistence import atomic_text
            atomic_text(path, '# AutoMorphTree numeric selection report\n\nSelection diagnostics, not independent test evidence.\n\n```json\n'
                        +json.dumps(self.report(), indent=2, allow_nan=False)+'\n```\n')
        else:
            write_json(path, self.report())

    def __getattr__(self, name):
        # Delegate only the public native interface; artifacts keep their native format.
        allowed = {'predict', 'predict_proba', 'score', 'save_model', 'save', 'export_python', 'export_bundle',
                   'save_report', 'inspect_algorithm', 'classes_', 'params_', 'n_features_in_', 'feature_names_in_'}
        model = self.__dict__.get('model_')
        if name in allowed and model is not None:
            return getattr(model, name)
        raise AttributeError(name)

    def inspect(self):
        return self.report()


def load_model(path, *, limits=None, backend='pure'):
    """Bounded dispatch on an explicit format; never infer a task from target values."""
    data = read_json(path, limits)
    cls = {'MorphTreeRegressionModel': MorphTreeRegressor, 'MorphTreeMulticlassModel': MorphTreeMulticlassClassifier}.get(
        data.get('format'), MorphTreeClassifier)
    return cls._from_payload(data, limits=limits, backend=backend)
