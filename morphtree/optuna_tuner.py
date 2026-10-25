"""Optional Optuna TPE, pruning, conditional search and context-checked resume."""
from __future__ import annotations

from dataclasses import asdict, replace
import math

from .config import VERSION, canonical_hash
from .tuning import bounded_parameter, decide_tuned_candidate, prediction_loss, tunable_parameters, tuning_record
from .control import RunControl, SearchInterrupted


def sampler_options(config):
    # Pin options whose defaults differ between Optuna 4 and 5.
    return dict(n_startup_trials=config.optuna_startup_trials, multivariate=False,
                constant_liar=False, n_ei_candidates=24)


def study_context(base, config, dataset, base_guard_limit):
    import optuna
    return canonical_hash({'dataset': dataset, 'genome': base.genome.to_dict(),
        'parameters': asdict(base.params), 'seed': config.seed,
        'early_stopping': config.early_stopping_rounds, 'version': VERSION,
        'metric': 'regression_rmse_v1' if getattr(base, 'objective', 'binary_logloss') == 'squared_error' else 'binary_logloss_v1',
        'search_space': {'version': 1, 'active': tunable_parameters(base.genome)},
        'guard_tolerance': config.guard_tolerance, 'global_guard_tolerance': config.global_guard_tolerance,
        'base_guard_limit': base_guard_limit, 'sampler': {'name': 'TPE', **sampler_options(config)},
        'optuna_major': int(optuna.__version__.split('.')[0])})


class OptunaTuner:
    def __init__(self):
        self.study = None
        self.context = None

    def optimize(self, base, evaluate, X_search, y_search, X_guard, y_guard, config, base_guard_limit):
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Optuna is optional: install morphtree[optuna] or choose tuner='none'") from exc
        before = prediction_loss(base, X_search, y_search)
        guard_before = prediction_loss(base, X_guard, y_guard)
        record = tuning_record(base.genome, before, guard_before)
        record["backend"] = "optuna"
        # Includes the runtime version, exact TRAIN/SEARCH contents, genome, baseline numbers and seed.
        context = study_context(base, config, self.context, base_guard_limit)
        control = RunControl(getattr(self, 'cancel', None), config.tune_timeout)
        storage = None
        try:
            if config.optuna_storage is not None:
                from sqlalchemy.pool import NullPool
                storage = optuna.storages.RDBStorage(config.optuna_storage, engine_kwargs={"poolclass": NullPool})
            self.study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed, **sampler_options(config)),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
                                        storage=storage, study_name=config.optuna_study_name or "MorphTree-" + context[:20],
                                        load_if_exists=config.optuna_storage is not None)
        except Exception:
            if storage is not None:
                storage.remove_session()
            raise RuntimeError('Unable to initialize Optuna storage or study; check private runtime settings') from None
        stored_context = self.study.user_attrs.get("morphtree_context")
        if stored_context is not None and stored_context != context:
            if storage is not None:
                storage.remove_session()
            raise ValueError("Optuna study context mismatch: do not mix runtime versions, datasets, genomes or "
                             "baseline parameters; use a new study name")
        if self.study.trials and stored_context is None:
            if storage is not None:
                storage.remove_session()
            raise ValueError("Refusing to reuse an existing study without MorphTree context")
        self.study.set_user_attr("morphtree_context", context)
        self.study.set_user_attr("structural_fingerprint", base.genome.fingerprint())
        active = tunable_parameters(base.genome)

        def objective(trial):
            control.check()
            values = {}
            for name in active:
                initial = getattr(base.params, name)
                if name in ("n_estimators", "max_leaves", "min_samples_leaf"):
                    low = bounded_parameter(name, initial * 0.5)
                    high = bounded_parameter(name, initial * 1.6)
                    values[name] = trial.suggest_int(name, low, max(low, high))
                else:
                    low = bounded_parameter(name, max(initial, 0.01) * 0.5)
                    high = bounded_parameter(name, max(initial, 0.01) * 1.8)
                    values[name] = trial.suggest_float(name, low, max(low, high), log=low > 0)
            params = replace(base.params, **values)

            def callback(iteration, loss):
                control.check()
                trial.report(loss, iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                return True

            candidate = evaluate(base.genome, params, callback=callback, early_stopping_rounds=config.early_stopping_rounds)
            if len(candidate.trees) < params.n_estimators:
                params = replace(params, n_estimators=len(candidate.trees))
                candidate = evaluate(base.genome, params, callback=lambda iteration, loss: control.check())
            if candidate.genome.fingerprint() != record["frozen_fingerprint"]:
                raise RuntimeError("Optuna attempted to change the frozen genome")
            trial.set_user_attr("effective_parameters", asdict(params))
            loss = prediction_loss(candidate, X_search, y_search)
            guard_loss = prediction_loss(candidate, X_guard, y_guard)
            if not math.isfinite(loss) or not math.isfinite(guard_loss):
                raise ArithmeticError('Nonfinite trial metric')
            feasible = guard_loss <= min(guard_before + config.guard_tolerance,
                                        base_guard_limit + config.global_guard_tolerance)
            trial.set_user_attr('guard_loss', guard_loss)
            trial.set_user_attr('feasible', feasible)
            return loss

        first_new_trial = len(self.study.trials)
        if first_new_trial:
            # RNG state is not persisted by Optuna. Avoid replaying identical startup proposals after reopening.
            self.study.sampler = optuna.samplers.TPESampler(seed=(config.seed + first_new_trial) % (2**32), **sampler_options(config))
        try:
            self.study.optimize(objective, n_trials=config.tune_trials, timeout=config.tune_timeout, n_jobs=1,
                                show_progress_bar=False, catch=(ArithmeticError, ValueError))
            control.check()
        except SearchInterrupted as exc:
            record['stop_reason'] = exc.reason
        complete = [trial for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE
                    and trial.value is not None and math.isfinite(trial.value) and "effective_parameters" in trial.user_attrs]
        feasible = [trial for trial in complete if trial.user_attrs.get('feasible') is True]
        record["trials"] = [{"index": trial.number, "state": trial.state.name, "search_loss": trial.value,
                              "parameters": trial.params, 'guard_loss': trial.user_attrs.get('guard_loss'),
                              'feasible': trial.user_attrs.get('feasible', False)} for trial in self.study.trials]
        record["new_trials"] = len(self.study.trials) - first_new_trial
        record['sampler'] = {'name': 'TPE', **sampler_options(config)}
        record['sampling_note'] = 'Startup uses random sampling; requested trials do not prove adaptive execution'
        record["study_context"] = context
        record["completed_trials"] = len(complete)
        record['feasible_trials'] = len(feasible)
        if feasible and 'stop_reason' not in record:
            best_trial = min(feasible, key=lambda trial: trial.value)
            try:
                control.check()
                candidate = evaluate(base.genome, replace(base.params, **best_trial.user_attrs["effective_parameters"]),
                                     callback=lambda iteration, loss: control.check())
            except SearchInterrupted as exc:
                record['stop_reason'] = exc.reason
                candidate = base
        else:
            candidate = base
            record["reason"] = "no_completed_trials"
        selected, record = decide_tuned_candidate(base, candidate, record, X_search, y_search, X_guard, y_guard, config, base_guard_limit)
        if not complete:
            record.update(completed=False, accepted=False, reason="no_completed_trials")
        elif not feasible:
            record.update(completed=True, accepted=False, reason='no_feasible_trials')
        if storage is not None:
            storage.remove_session()
        return selected, record
