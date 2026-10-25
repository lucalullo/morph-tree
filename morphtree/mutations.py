"""Error-guided structural programs, novelty memory and local plateau protocol."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import random

from .config import POLICIES
from .diagnostics import ErrorProfiler
from .metrics import logloss, paired_improvement
from .control import RunControl, SearchInterrupted


@dataclass(frozen=True)
class Mutation:
    genome: object
    gene: str
    reason: str
    relevance: float


def propose_mutations(genome, profile):
    strengths = {item["name"]: item["strength"] * item["confidence"] for item in profile["diagnoses"]}
    mapping = {
        "growth_policy": ("INTERACTION_LIKE_ERROR", "SMALL_UNSTABLE_REGION"),
        "binning_policy": ("LOCAL_ERROR_CONCENTRATION", "SIGNED_BIAS"),
        "missing_policy": ("MISSING_RELATED_ERROR",),
        "leaf_policy": ("HIGH_UNCERTAINTY", "SIGNED_BIAS", "NOISE_LIKE"),
        "regularization_policy": ("SMALL_UNSTABLE_REGION", "NOISE_LIKE"),
        "sample_policy": ("PERSISTENT_ERROR", "IMBALANCE_RELATED_ERROR"),
        "error_policy": ("PERSISTENT_ERROR", "IMBALANCE_RELATED_ERROR"),
        "local_policy": ("LOCAL_ERROR_CONCENTRATION", "MISSING_RELATED_ERROR"),
        "update_policy": ("NOISE_LIKE", "HIGH_RESIDUAL_VARIANCE"),
    }
    proposals = []
    for gene, values in POLICIES.items():
        diagnosis = max(mapping[gene], key=lambda key: strengths.get(key, 0))
        relevance = 0.02 + strengths.get(diagnosis, 0)
        for value in values:
            if value != getattr(genome, gene):
                proposals.append(Mutation(genome.mutate(**{gene: value}), gene,
                                          f"Hypothesis {diagnosis}: change {gene} to {value}", relevance))
    programs = [
        ("OBJECTIVE_GAIN",),
        ("OBJECTIVE_GAIN", "BALANCE", "MUL"),
        ("OBJECTIVE_GAIN", "HESSIAN_BALANCE", "MUL"),
        ("OBJECTIVE_GAIN", "RESIDUAL_VARIANCE_GAIN", "MUL_SLOT:0", "ADD"),
        ("OBJECTIVE_GAIN", "UNCERTAINTY_GAIN", "MUL_SLOT:1", "ADD"),
    ]
    # Construct new bounded expressions from the incumbent, rather than only selecting templates.
    for signal, slot in (("RESIDUAL_VARIANCE_GAIN", 0), ("UNCERTAINTY_GAIN", 1)):
        if signal not in genome.split_program and len(genome.split_program) + 3 <= 15:
            programs.append(genome.split_program + (signal, f"MUL_SLOT:{slot}", "ADD"))
    if "BALANCE" not in genome.split_program and len(genome.split_program) + 2 <= 15:
        programs.append(genome.split_program + ("BALANCE", "MUL"))
    for program in programs:
        if program != genome.split_program:
            reason = "HIGH_RESIDUAL_VARIANCE" if "RESIDUAL_VARIANCE_GAIN" in program else "HIGH_UNCERTAINTY"
            proposals.append(Mutation(genome.mutate(split_program=program), "split_program",
                                      f"Hypothesis {reason}: construct bounded split program", 0.03 + strengths.get(reason, 0)))
    unique = {p.genome.fingerprint(): p for p in proposals}
    return sorted(unique.values(), key=lambda p: (-p.relevance, p.gene, p.genome.fingerprint()))


class StructuralSearch:
    """A plateau is relative to the fully evaluated current neighborhood and fixed numbers."""
    def __init__(self, config):
        self.config = config
        self.control = RunControl(timeout=config.search_timeout)

    def run(self, initial, evaluate, X_train, y_train, X_search, y_search, X_guard, y_guard):
        incumbent = initial
        base_guard = logloss(y_guard, incumbent.predict_proba(X_guard))
        history, profiles, seen = [], [], {}
        final_coverage = {}
        stalled = 0
        stop_reason = "budget_exhausted"
        plateau = False
        for round_index in range(1, self.config.structural_rounds + 1):
            try:
                self.control.check()
            except SearchInterrupted as exc:
                stop_reason = exc.reason
                break
            genome = incumbent.genome
            parent = genome.fingerprint()
            old_search_p = incumbent.predict_proba(X_search)
            old_search = logloss(y_search, old_search_p)
            old_guard = logloss(y_guard, incumbent.predict_proba(X_guard))
            if parent not in seen:
                profile = ErrorProfiler().profile(X_train, y_train, incumbent.memory_, X_search, y_search, old_search_p)
                proposals = propose_mutations(genome, profile)
                if not self.config.diagnostic_guidance:
                    random.Random(str(self.config.seed) + parent).shuffle(proposals)
                excluded = []
                if incumbent.params.row_sample_count(len(y_train)) == len(y_train):
                    active = []
                    for proposal in proposals:
                        only_sampling = proposal.genome.mutate(sample_policy=genome.sample_policy) == genome
                        if proposal.gene == "sample_policy" and only_sampling:
                            excluded.append({"gene": proposal.gene, "fingerprint": proposal.genome.fingerprint(),
                                             "reason": "all_training_rows_selected_at_fixed_parameters"})
                        else:
                            active.append(proposal)
                    proposals = active
                seen[parent] = {"profile": profile, "proposals": proposals, "excluded": excluded,
                                "full": set(), "screened": set()}
                profiles.append({"fingerprint": parent, "excluded_inactive_mutations": excluded, **profile})
            state = seen[parent]
            proposals = state["proposals"]
            untried = [p for p in proposals if p.genome.fingerprint() not in state["full"] | state["screened"]]
            # Screened candidates are deferred, not declared rejected. All must be fully checked for a plateau.
            deferred = [p for p in proposals if p.genome.fingerprint() in state["screened"]]
            queue = untried if untried else deferred
            width = self.config.structural_mutants
            if state["profile"]["confidence"] < 0.5:
                width = min(width, 2)
            candidates = queue[:width]
            if not candidates:
                plateau, stop_reason = True, "verified_local_plateau"
                final_coverage = {"available": len(proposals), "fully_evaluated": len(state["full"]), "screened_pending": 0,
                                  "excluded_inactive": state["excluded"]}
                break
            winner = None
            for proposal in candidates:
                if self.config.structural_evaluations is not None and len(history) >= self.config.structural_evaluations:
                    break
                candidate_start = time.perf_counter()
                child_fp = proposal.genome.fingerprint()
                is_confirmation = child_fp in state["screened"]
                probe_step = max(2, int(self.config.probe_fraction * incumbent.params.n_estimators))
                can_probe = not is_confirmation and self.config.probe_fraction > 0 and probe_step < incumbent.params.n_estimators
                did_screen = False

                def probe(iteration, loss):
                    nonlocal did_screen
                    self.control.check()
                    if can_probe and iteration == probe_step and len(incumbent.search_curve_) >= iteration:
                        if loss > incumbent.search_curve_[iteration - 1] + self.config.probe_margin:
                            did_screen = True
                            return False
                    return True

                try:
                    self.control.check()
                    child = evaluate(proposal.genome, incumbent.params, callback=probe)
                except SearchInterrupted as exc:
                    stop_reason = exc.reason
                    history.append({'round': round_index, 'parent_fingerprint': parent, 'child_fingerprint': child_fp,
                                    'gene': proposal.gene, 'reason': proposal.reason, 'stage': 'interrupted',
                                    'decision': exc.reason, 'accepted': False,
                                    'cost_seconds': time.perf_counter() - candidate_start})
                    break
                record = {"round": round_index, "parent_fingerprint": parent, "child_fingerprint": child_fp,
                          "gene": proposal.gene, "reason": proposal.reason, "relevance": proposal.relevance,
                          "stage": "confirmation" if is_confirmation else "full", "accepted": False,
                          "params_unchanged": child.params == initial.params}
                record['cost_seconds'] = time.perf_counter() - candidate_start
                record['trees_trained'] = len(child.trees) if hasattr(child, 'trees') else None
                if did_screen:
                    state["screened"].add(child_fp)
                    record.update(stage="cheap_probe", search_loss=child.search_curve_[-1], guard_loss=None,
                                  decision="deferred_for_full_check")
                    history.append(record)
                    continue
                state["screened"].discard(child_fp)
                state["full"].add(child_fp)
                child_p = child.predict_proba(X_search)
                search_loss = logloss(y_search, child_p)
                guard_loss = logloss(y_guard, child.predict_proba(X_guard))
                improvement, lower, se = paired_improvement(y_search, old_search_p, child_p, self.config.confidence_z)
                penalty = self.config.complexity_penalty * max(0, proposal.genome.complexity() - genome.complexity())
                threshold = self.config.min_morph_improvement + penalty
                qualifies = lower > threshold and guard_loss <= min(old_guard + self.config.guard_tolerance,
                                                                   base_guard + self.config.global_guard_tolerance)
                record.update(search_loss=search_loss, guard_loss=guard_loss, search_delta=improvement,
                              lower_gain=lower, gain_standard_error=se, guard_delta=old_guard - guard_loss,
                              decision="eligible_not_selected" if qualifies else "rejected")
                history.append(record)
                if qualifies and (winner is None or search_loss + penalty < winner[0]):
                    winner = (search_loss + penalty, child, record)
            if winner:
                incumbent = winner[1]
                winner[2].update(accepted=True, decision="committed")
                stalled = 0
            else:
                stalled += 1
            final_coverage = {"available": len(proposals), "fully_evaluated": len(state["full"]),
                              "screened_pending": len(state["screened"]), "excluded_inactive": state["excluded"]}
            if stop_reason in ('cancelled', 'timeout'):
                break
            exhausted = len(state["full"]) == len(proposals) and not state["screened"]
            required_stalls = min(self.config.structural_patience, max(1, math.ceil(len(proposals) / width)))
            if winner is None and exhausted and stalled >= required_stalls:
                plateau, stop_reason = True, "verified_local_plateau"
                break
            if self.config.structural_evaluations is not None and len(history) >= self.config.structural_evaluations:
                break
        if not plateau and history and history[-1]["parent_fingerprint"] != incumbent.genome.fingerprint():
            final_coverage = {"available": None, "fully_evaluated": 0, "screened_pending": 0,
                              "note": "new incumbent has not yet been explored"}
        return incumbent, {"stop_reason": stop_reason, "plateau_reached": plateau, "history": history,
                           "profiles": profiles, "coverage": final_coverage,
                           "scope": "active one-step neighborhood; reported no-op exclusions; fixed HyperParams; current TRAIN/SEARCH/GUARD"}
