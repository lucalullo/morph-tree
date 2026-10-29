"""Small deterministic, conditional native tuner; never changes structural genes."""
from __future__ import annotations

from dataclasses import asdict, replace
import time
import math

from .metrics import logloss, rmse
from .control import RunControl, SearchInterrupted


TUNABLE_ORDER = ("learning_rate", "l2", "min_samples_leaf", "max_leaves", "n_estimators", "feature_fraction", "row_fraction",
                 "path_smooth", "max_delta_step", "depth_reg_strength", "error_weight", "update_decay", "gate_strength",
                 "signal_slot_0", "signal_slot_1")


def tunable_parameters(genome):
    active = set(genome.active_parameters())
    return tuple(name for name in TUNABLE_ORDER if name in active)


def bounded_parameter(name, value):
    integer_limits = {"min_samples_leaf": (1, 10**9), "max_leaves": (2, 4096), "n_estimators": (1, 4096)}
    if name in integer_limits:
        low, high = integer_limits[name]
        return min(high, max(low, int(round(value))))
    if name in ("feature_fraction", "row_fraction", "learning_rate"):
        return min(1.0, max(0.001, value))
    if name.startswith("signal_slot_"):
        return min(10.0, max(0.0, value))
    return max(1e-6, min(1e6, value))


def tuning_record(genome, base_search, base_guard):
    return {"attempted": True, "backend": "native", "completed": False, "accepted": False,
            "reason": "no_completed_trials", "trials": [], "active_parameters": list(genome.active_parameters()),
            "tunable_parameters": list(tunable_parameters(genome)), "frozen_fingerprint": genome.fingerprint(),
            "search_loss_before": base_search, "guard_loss_before": base_guard}


def prediction_loss(model, X, y):
    if getattr(model, 'objective', 'binary_logloss') == 'squared_error':
        return rmse(y, model.predict_raw(X))
    return logloss(y, model.predict_proba(X))


def decide_tuned_candidate(base, candidate, record, X_search, y_search, X_guard, y_guard, config, base_guard_limit):
    record["completed"] = bool(record["trials"])
    record["selected_parameters"] = asdict(candidate.params)
    new_search = prediction_loss(candidate, X_search, y_search)
    new_guard = prediction_loss(candidate, X_guard, y_guard)
    record.update(search_loss_after=new_search, guard_loss_after=new_guard)
    floor = 0.0 if getattr(base, 'objective', 'binary_logloss') == 'squared_error' else 1e-12
    passes = math.isfinite(new_search) and math.isfinite(new_guard) and new_search < record["search_loss_before"] - floor and new_guard <= min(
        record["guard_loss_before"] + config.guard_tolerance, base_guard_limit + config.global_guard_tolerance)
    if passes:
        record.update(accepted=True, reason="accepted")
        result = candidate
    else:
        record["reason"] = "rejected_or_no_search_gain" if record["completed"] else "no_completed_trials"
        result = base
    record["effective_parameters"] = asdict(result.params)
    assert result.genome.fingerprint() == record["frozen_fingerprint"]
    return result, record


class MorphTuner:
    """Coarse coordinate probes, sensitivity ranking and local refinement."""
    def optimize(self, base, evaluate, X_search, y_search, X_guard, y_guard, config, base_guard_limit):
        start = time.monotonic()
        before = logloss(y_search, base.predict_proba(X_search))
        guard = logloss(y_guard, base.predict_proba(X_guard))
        record = tuning_record(base.genome, before, guard)
        control = RunControl(getattr(self, 'cancel', None), config.tune_timeout)
        best, best_loss = base, before
        tried, sensitivity = set(), {}
        stale = 0
        proposal_list = []
        for name in tunable_parameters(base.genome):
            value = getattr(base.params, name)
            if value == 0:
                value = 0.1
            for factor in (0.7, 1.3):
                proposal_list.append((name, bounded_parameter(name, value * factor), "coarse"))
        # Reserve about half the budget for sensitivity-guided refinements.
        coarse_budget = max(2, (config.tune_trials + 1) // 2)
        for trial_index in range(config.tune_trials):
            try:
                control.check()
            except SearchInterrupted as exc:
                record['stop_reason'] = exc.reason
                break
            if config.tune_timeout and time.monotonic() - start >= config.tune_timeout:
                record["stop_reason"] = "timeout"
                break
            if trial_index < coarse_budget and proposal_list:
                name, value, phase = proposal_list.pop(0)
            else:
                if sensitivity:
                    ranked = sorted(sensitivity, key=lambda k: (-sensitivity[k], k))
                    name = ranked[((trial_index - coarse_budget) // 2) % len(ranked)]
                else:
                    name = tunable_parameters(base.genome)[(trial_index // 2) % len(tunable_parameters(base.genome))]
                factor = 0.85 if trial_index % 2 == 0 else 1.15
                value, phase = bounded_parameter(name, max(getattr(best.params, name), 1e-6) * factor), "local"
            params = replace(best.params, **{name: value})
            key = tuple(asdict(params).items())
            if key in tried or params == best.params:
                continue
            tried.add(key)
            def callback(iteration, loss):
                control.check()
                return True
            try:
                candidate = evaluate(base.genome, params, callback=callback, early_stopping_rounds=config.early_stopping_rounds)
            except SearchInterrupted as exc:
                record['stop_reason'] = exc.reason
                break
            except (ArithmeticError, ValueError) as exc:
                record['trials'].append({'index': trial_index, 'state': 'FAIL', 'reason': type(exc).__name__})
                continue
            # n_estimators is a numerical parameter: choosing a retained prefix is allowed only here.
            if len(candidate.trees) < params.n_estimators:
                params = replace(params, n_estimators=len(candidate.trees))
                try:
                    candidate = evaluate(base.genome, params, callback=callback)
                except SearchInterrupted as exc:
                    record['stop_reason'] = exc.reason
                    break
            assert candidate.genome.fingerprint() == record["frozen_fingerprint"]
            loss = logloss(y_search, candidate.predict_proba(X_search))
            candidate_guard = logloss(y_guard, candidate.predict_proba(X_guard))
            feasible = math.isfinite(loss) and math.isfinite(candidate_guard) and candidate_guard <= min(
                logloss(y_guard, best.predict_proba(X_guard)) + config.guard_tolerance,
                guard + config.guard_tolerance, base_guard_limit + config.global_guard_tolerance)
            gain = best_loss - loss
            sensitivity[name] = max(sensitivity.get(name, 0.0), abs(gain))
            record["trials"].append({"index": trial_index, "phase": phase, "changed_parameter": name,
                                     "value": value, "search_loss": loss, "guard_loss": candidate_guard,
                                     "feasible": feasible, "parameters": asdict(candidate.params)})
            if feasible and loss < best_loss - 1e-12:
                best, best_loss, stale = candidate, loss, 0
            else:
                stale += 1
            if stale >= config.tune_patience and trial_index + 1 >= coarse_budget:
                record["stop_reason"] = "local_patience"
                break
        record.setdefault("stop_reason", "budget_exhausted")
        record["sensitivity"] = sensitivity
        return decide_tuned_candidate(base, best, record, X_search, y_search, X_guard, y_guard, config, base_guard_limit)
