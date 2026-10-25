"""Immutable structural genes, numerical parameters and protocol settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from numbers import Real

VERSION = "0.6.0.dev0"
SPEC_VERSION = 1
POLICIES = {
    "growth_policy": ("leafwise", "depthwise", "lookahead"),
    "binning_policy": ("quantile", "uniform"),
    "missing_policy": ("learned", "left", "right"),
    "leaf_policy": ("newton", "smoothed", "clipped", "uncertainty"),
    "regularization_policy": ("global", "depth_scaled"),
    "sample_policy": ("uniform", "hardness", "uncertainty"),
    "error_policy": ("none", "persistent_weighting"),
    "local_policy": ("none", "low_support", "missing_heavy"),
    "update_policy": ("constant", "decay"),
}
SIGNALS = ("OBJECTIVE_GAIN", "RESIDUAL_VARIANCE_GAIN", "BALANCE", "HESSIAN_BALANCE", "UNCERTAINTY_GAIN")


def checked_number(name, value, low=0.0, high=None, *, strict=False):
    """Validate the actual float used by the runtime, not a wider external scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a finite Python float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or (strict and value == low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _set_number(instance, name, low=0.0, high=None, *, strict=False):
    value = checked_number(name, getattr(instance, name), low, high, strict=strict)
    object.__setattr__(instance, name, value)


def checked_int(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def canonical_hash(data):
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_program(program):
    if not isinstance(program, (list, tuple)) or not 1 <= len(program) <= 15:
        raise ValueError("A split program needs 1..15 instructions")
    depth = 0
    for token in program:
        if not isinstance(token, str):
            raise ValueError("Program instructions must be strings")
        if token in SIGNALS:
            depth += 1
        elif token in ("MUL_SLOT:0", "MUL_SLOT:1"):
            if depth < 1:
                raise ValueError("Split program stack underflow")
        elif token in ("ADD", "MUL", "MIN", "MAX"):
            if depth < 2:
                raise ValueError("Split program stack underflow")
            depth -= 1
        else:
            raise ValueError(f"Unknown split instruction: {token}")
    if depth != 1 or "OBJECTIVE_GAIN" not in program:
        raise ValueError("Split program must return one score and include OBJECTIVE_GAIN")


@dataclass(frozen=True)
class AlgorithmGenome:
    growth_policy: str = "leafwise"
    binning_policy: str = "quantile"
    missing_policy: str = "learned"
    leaf_policy: str = "newton"
    regularization_policy: str = "global"
    sample_policy: str = "uniform"
    error_policy: str = "none"
    local_policy: str = "none"
    update_policy: str = "constant"
    split_program: tuple[str, ...] = ("OBJECTIVE_GAIN",)

    def __post_init__(self):
        for name, choices in POLICIES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"Invalid {name}")
        validate_program(self.split_program)
        object.__setattr__(self, "split_program", tuple(self.split_program))

    def mutate(self, **changes):
        return replace(self, **changes)

    def to_dict(self):
        data = asdict(self)
        data["split_program"] = list(self.split_program)
        return data

    def fingerprint(self):
        return canonical_hash({"spec_version": SPEC_VERSION, "genome": self.to_dict()})

    def active_parameters(self):
        names = ["learning_rate", "n_estimators", "max_bins", "max_leaves", "max_depth", "min_samples_leaf",
                 "l1", "l2", "min_hessian", "min_gain", "feature_fraction", "row_fraction"]
        conditional = {
            "path_smooth": self.leaf_policy == "smoothed",
            "max_delta_step": self.leaf_policy == "clipped",
            "depth_reg_strength": self.regularization_policy == "depth_scaled",
            "error_memory_beta": self.error_policy != "none" or self.sample_policy != "uniform",
            "error_weight": self.error_policy != "none",
            "update_decay": self.update_policy == "decay",
            "gate_strength": self.local_policy != "none" or self.leaf_policy == "uncertainty",
        }
        names.extend(name for name, active in conditional.items() if active)
        names.extend(f"signal_slot_{i}" for i in range(2) if f"MUL_SLOT:{i}" in self.split_program)
        return tuple(names)

    def complexity(self):
        base = AlgorithmGenome()
        return sum(getattr(self, k) != getattr(base, k) for k in POLICIES) + len(self.split_program) - 1


@dataclass(frozen=True)
class HyperParams:
    learning_rate: float = 0.1
    n_estimators: int = 32
    max_bins: int = 31
    max_leaves: int = 7
    max_depth: int = 5
    min_samples_leaf: int = 6
    l1: float = 0.0
    l2: float = 2.0
    min_hessian: float = 1e-3
    min_gain: float = 1e-5
    feature_fraction: float = 1.0
    row_fraction: float = 0.8
    path_smooth: float = 8.0
    max_delta_step: float = 2.0
    depth_reg_strength: float = 0.2
    error_memory_beta: float = 0.9
    error_weight: float = 1.0
    update_decay: float = 0.75
    gate_strength: float = 1.0
    signal_slot_0: float = 0.25
    signal_slot_1: float = 0.25

    def __post_init__(self):
        for name, low, high in (("n_estimators", 1, 4096), ("max_bins", 2, 4096), ("max_leaves", 2, 4096),
                                ("max_depth", 1, 128), ("min_samples_leaf", 1, 10**9)):
            checked_int(name, getattr(self, name), low, high)
        _set_number(self, "learning_rate", 0, 1, strict=True)
        _set_number(self, "min_hessian", 0, 1e6, strict=True)
        for name in ("l1", "l2", "min_gain", "path_smooth", "depth_reg_strength", "error_weight", "update_decay", "gate_strength"):
            _set_number(self, name, 0, 1e6)
        _set_number(self, "max_delta_step", 0, 1e6, strict=True)
        for name in ("feature_fraction", "row_fraction"):
            _set_number(self, name, 0, 1, strict=True)
        _set_number(self, "error_memory_beta", 0, 1)
        if self.error_memory_beta >= 1:
            raise ValueError("error_memory_beta must be less than 1")
        for name in ("signal_slot_0", "signal_slot_1"):
            _set_number(self, name, 0, 10)

    def row_sample_count(self, n_rows):
        """Fixed-size training subsample; leaf support can require the full dataset."""
        if type(n_rows) is not int or n_rows < 1:
            raise ValueError("n_rows must be a positive Python integer")
        return min(n_rows, max(2 * self.min_samples_leaf, round(n_rows * self.row_fraction)))


@dataclass(frozen=True)
class MorphTreeConfig:
    seed: int = 42
    search_fraction: float = 0.15
    guard_fraction: float = 0.15
    enable_morphing: bool = False
    enable_tuning: bool = True
    structural_rounds: int = 16
    structural_mutants: int = 3
    structural_patience: int = 2
    structural_evaluations: int | None = None
    search_timeout: float | None = None
    freeze_on_budget: bool = False
    diagnostic_guidance: bool = False
    memory_enabled: bool = False
    min_morph_improvement: float = 0.0002
    guard_tolerance: float = 0.001
    global_guard_tolerance: float = 0.002
    complexity_penalty: float = 0.00001
    confidence_z: float = 0.0
    probe_fraction: float = 0.35
    probe_margin: float = 0.04
    tuner: str = "none"
    tune_trials: int = 16
    optuna_startup_trials: int = 5
    tune_patience: int = 6
    tune_timeout: float | None = None
    early_stopping_rounds: int = 6
    optuna_storage: str | None = None
    optuna_study_name: str | None = None
    refit_full: bool = True

    def __post_init__(self):
        for name in ("enable_morphing", "enable_tuning", "refit_full", "freeze_on_budget", "diagnostic_guidance", "memory_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name, low, high in (("seed", 0, 2**32 - 1), ("structural_rounds", 0, 10000),
                                ("structural_mutants", 1, 128), ("structural_patience", 1, 100),
                                ("tune_trials", 1, 100000), ("tune_patience", 1, 10000),
                                ("optuna_startup_trials", 1, 100000),
                                ("early_stopping_rounds", 0, 4096)):
            checked_int(name, getattr(self, name), low, high)
        for name in ("search_fraction", "guard_fraction"):
            _set_number(self, name, 0, 0.5, strict=True)
        if self.search_fraction + self.guard_fraction >= 0.6:
            raise ValueError("SEARCH and GUARD must leave more than 40% for TRAIN")
        for name in ("min_morph_improvement", "guard_tolerance", "global_guard_tolerance", "complexity_penalty", "probe_margin"):
            _set_number(self, name, 0, 10)
        _set_number(self, "confidence_z", 0, 5)
        _set_number(self, "probe_fraction", 0, 1)
        if self.tune_timeout is not None:
            _set_number(self, "tune_timeout", 0, 1e9, strict=True)
        if self.search_timeout is not None:
            _set_number(self, "search_timeout", 0, 1e9, strict=True)
        if self.structural_evaluations is not None:
            checked_int("structural_evaluations", self.structural_evaluations, 0, 1000000)
        if self.tuner not in ("none", "native", "optuna"):
            raise ValueError("tuner must be none, native or optuna")
        for name in ("optuna_storage", "optuna_study_name"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")


# Versioned experimental capability; consolidated config.py is unchanged.
TASK_GROWTH_FORMATS = {'regression':'MorphTreeRegressionGrowthV2',
                       'multiclass':'MorphTreeMulticlassGrowthV2'}

def validate_task_genome(task, genome):
    if task not in ('binary','multiclass','regression') or not isinstance(genome,AlgorithmGenome):
        raise ValueError('Known task and native AlgorithmGenome required')
    if task!='binary' and genome!=AlgorithmGenome(growth_policy=genome.growth_policy):
        raise ValueError('Only Base/depthwise growth is admitted for this nonbinary capability')
    return genome

def decode_task_genome(task, payload):
    if payload.get('format')!=TASK_GROWTH_FORMATS[task]:
        if 'genome' in payload:raise ValueError('Legacy native format cannot acquire a genome field')
        return AlgorithmGenome()
    if type(payload.get('genome')) is not dict:
        raise ValueError('V2 native growth requires an explicit genome')
    try:genome=validate_task_genome(task,AlgorithmGenome(**payload['genome']))
    except (TypeError,ValueError):raise ValueError('Invalid task-compatible V2 genome') from None
    if genome==AlgorithmGenome():raise ValueError('Base retains its legacy native format')
    return genome
