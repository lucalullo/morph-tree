"""Persistent sample-error memory and bounded, explicitly heuristic diagnosis."""
from __future__ import annotations
from array import array
from dataclasses import dataclass
import math

class ErrorMemory:

    def __init__(self, size, beta=0.9):
        self.beta = beta
        self.ema_error = array('d', [0.0]) * size
        self.signed_residual_ema = array('d', [0.0]) * size
        self.wrong_streak = array('I', [0]) * size
        self.uncertainty = array('d', [0.0]) * size
        self.hardness = array('d', [0.0]) * size
        self.direction_consistency = array('d', [0.0]) * size
        self.updates = 0

    def update(self, y, p):
        if len(y) != len(self.ema_error) or len(p) != len(y):
            raise ValueError('ErrorMemory sample alignment mismatch')
        self.updates += 1
        correction = 1 - self.beta ** self.updates
        for i, (label, probability) in enumerate(zip(y, p)):
            residual = label - probability
            self.ema_error[i] = self.beta * self.ema_error[i] + (1 - self.beta) * abs(residual)
            self.signed_residual_ema[i] = self.beta * self.signed_residual_ema[i] + (1 - self.beta) * residual
            self.wrong_streak[i] = min(2 ** 32 - 1, self.wrong_streak[i] + 1) if (probability >= 0.5) != label else 0
            self.uncertainty[i] = 4 * probability * (1 - probability)
            mean_error = self.ema_error[i] / max(correction, 1e-12)
            self.direction_consistency[i] = abs(self.signed_residual_ema[i]) / max(self.ema_error[i], 1e-12)
            self.hardness[i] = min(1.0, 0.7 * mean_error + 0.3 * min(self.wrong_streak[i] / 8, 1.0))

    def summary(self):
        n = max(1, len(self.ema_error))
        return {'updates': self.updates, 'mean_error_ema': math.fsum(self.ema_error) / n, 'mean_hardness': math.fsum(self.hardness) / n, 'persistent_fraction': sum((v >= 3 for v in self.wrong_streak)) / n, 'mean_uncertainty': math.fsum(self.uncertainty) / n, 'mean_direction_consistency': math.fsum(self.direction_consistency) / n, 'max_wrong_streak': max(self.wrong_streak, default=0)}

@dataclass(frozen=True)
class Diagnosis:
    name: str
    strength: float
    confidence: float
    evidence: str

class ErrorProfiler:
    """Diagnostics are hypotheses, not identified causes of prediction errors."""

    def profile(self, X_train, y_train, memory, X_search, y_search, p_search):
        if memory is None or len(memory.ema_error) != len(y_train):
            raise ValueError('Profiler requires aligned persistent TRAIN error memory')
        n = len(y_search)
        residuals = [label - p for label, p in zip(y_search, p_search)]
        absolute = [abs(r) for r in residuals]
        mean = math.fsum(absolute) / n
        bias = abs(math.fsum(residuals) / n)
        variance = math.fsum((r * r for r in residuals)) / n - (math.fsum(residuals) / n) ** 2
        memory_stats = memory.summary()
        confidence = min(1.0, math.sqrt(n / 120))
        groups = []
        local_gap, missing_gap, small_gap = (0.0, 0.0, 0.0)
        partition_bits = []
        for j in range(len(X_train[0])):
            present = sorted((row[j] for row in X_train if not math.isnan(row[j])))
            threshold = 0.5 * present[(len(present) - 1) // 2] + 0.5 * present[len(present) // 2] if present else 0.0
            low = [i for i, row in enumerate(X_search) if not math.isnan(row[j]) and row[j] <= threshold]
            high = [i for i, row in enumerate(X_search) if not math.isnan(row[j]) and row[j] > threshold]
            missing = [i for i, row in enumerate(X_search) if math.isnan(row[j])]
            partition_bits.append([not math.isnan(row[j]) and row[j] > threshold for row in X_search])
            for name, ids in (('low', low), ('high', high), ('missing', missing)):
                if len(ids) < 3:
                    continue
                group_error = math.fsum((absolute[i] for i in ids)) / len(ids)
                gap = max(0.0, group_error - mean)
                support = min(1.0, len(ids) / 20)
                local_gap = max(local_gap, gap * support)
                if name == 'missing':
                    missing_gap = max(missing_gap, gap * support)
                if len(ids) < max(8, n * 0.1):
                    small_gap = max(small_gap, gap * support)
                groups.append({'feature': j, 'region': name, 'count': len(ids), 'error': group_error, 'excess_error': gap})
        class_errors = []
        for label in (0, 1):
            ids = [i for i, value in enumerate(y_search) if value == label]
            class_errors.append(math.fsum((absolute[i] for i in ids)) / max(len(ids), 1))
        imbalance = min(y_train.count(0), y_train.count(1)) / len(y_train)
        interaction = 0.0
        for a in range(min(4, len(partition_bits))):
            for b in range(a + 1, min(4, len(partition_bits))):
                for abit, bbit in ((False, False), (False, True), (True, False), (True, True)):
                    ids = [i for i in range(n) if partition_bits[a][i] == abit and partition_bits[b][i] == bbit]
                    if len(ids) >= 5:
                        signed = abs(math.fsum((residuals[i] for i in ids)) / len(ids))
                        interaction = max(interaction, max(0.0, signed - bias - local_gap) * min(1, len(ids) / 20))
        strengths = {'PERSISTENT_ERROR': memory_stats['persistent_fraction'], 'HIGH_RESIDUAL_VARIANCE': min(1, variance * 4), 'SIGNED_BIAS': min(1, bias * 4), 'HIGH_UNCERTAINTY': memory_stats['mean_uncertainty'], 'LOCAL_ERROR_CONCENTRATION': min(1, local_gap * 4), 'SMALL_UNSTABLE_REGION': min(1, small_gap * 4), 'MISSING_RELATED_ERROR': min(1, missing_gap * 4), 'IMBALANCE_RELATED_ERROR': min(1, abs(class_errors[0] - class_errors[1]) * 3) if imbalance < 0.35 else 0.0, 'INTERACTION_LIKE_ERROR': min(1, interaction * 3), 'NOISE_LIKE': min(1, mean * 2) * max(0, 1 - local_gap * 5 - bias * 4 - interaction * 3), 'LOW_STRUCTURED_ERROR': max(0, 1 - max(local_gap * 4, missing_gap * 4, bias * 4, interaction * 3))}
        diagnoses = [Diagnosis(name, float(strength), confidence, 'TRAIN persistent memory and SEARCH residual partitions') for name, strength in strengths.items()]
        return {'diagnoses': [d.__dict__ for d in diagnoses], 'memory': memory_stats, 'regions': sorted(groups, key=lambda g: g['excess_error'], reverse=True)[:8], 'search_mean_absolute_residual': mean, 'search_signed_bias': bias, 'confidence': confidence, 'interpretation': 'heuristic hypotheses, not causal conclusions'}

    def profile_task(self, X, y, prediction, task):
        """Current-parent TRAIN-OOF residual hypotheses, not predictive utility.

        Multiclass uses its true K-vector residuals, not binary training.
        Regression stays continuous and is scaled using TRAIN targets only.
        The bounded region probe follows the existing profiler's logic.
        """
        n = len(y)
        if task not in ('multiclass', 'regression') or not n or len(X) != n or (len(prediction) != n):
            raise ValueError('Aligned task-specific residual data required')
        if task == 'regression':
            if any((type(v) not in (int, float) or not math.isfinite(v) for v in list(y) + list(prediction))):
                raise ValueError('Finite continuous targets/predictions required')
            mean = math.fsum((v / n for v in y))
            scale = math.hypot(*(v - mean for v in y)) / math.sqrt(n)
            scale = scale if scale > 1e-12 else 1.0
            vectors = [[(a - b) / scale for a, b in zip(y, prediction)]]
            labels = [None]
        else:
            labels = sorted(set(y))
            k = len(labels)
            scale = 1.0
            if labels != list(range(k)) or k < 2 or any((type(p) is not list or len(p) != k or any((type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in p)) or (abs(math.fsum(p) - 1) > 1e-12) for p in prediction)):
                raise ValueError('Native contiguous K-class normalized probabilities required')
            vectors = [[int(label == c) - p[c] for label, p in zip(y, prediction)] for c in labels]
        probes = [self._task_residual_probe(X, r) for r in vectors]
        names = ('INTERACTION_LIKE_ERROR', 'SMALL_UNSTABLE_REGION')
        confidence = min(1.0, math.sqrt(n / 120.0))
        return {'task': task, 'rows': n, 'target_scale_TRAIN_only': scale, 'diagnoses': [{'name': name, 'strength': max((p[name] for p in probes)), 'confidence': confidence, 'evidence': 'Current-parent TRAIN-OOF residual partitions'} for name in names], 'per_output': [{'class': c, 'probe': p} for c, p in zip(labels, probes)], 'policy_id': 'PRODUCT002_TASK_OOF_HEURISTIC_V1', 'fold_stability': 'UNKNOWN_NOT_ESTIMATED_FROM_ONE_STREAM', 'interpretation': 'Bounded residual associations; no causal or prospective utility claim'}

    def _task_residual_probe(self, X, residual):
        n = len(residual)
        absolute = [abs(v) for v in residual]
        if any((not math.isfinite(v) for v in absolute)):
            raise ValueError('Residual overflow')
        mean = math.fsum(absolute) / n
        bias = abs(math.fsum(residual) / n)
        local = small = interaction = 0.0
        bits = []
        for j in range(len(X[0])):
            present = sorted((row[j] for row in X if not math.isnan(row[j])))
            threshold = 0.5 * present[(len(present) - 1) // 2] + 0.5 * present[len(present) // 2] if present else 0.0
            low = [i for i, row in enumerate(X) if not math.isnan(row[j]) and row[j] <= threshold]
            high = [i for i, row in enumerate(X) if not math.isnan(row[j]) and row[j] > threshold]
            missing = [i for i, row in enumerate(X) if math.isnan(row[j])]
            bits.append([not math.isnan(row[j]) and row[j] > threshold for row in X])
            for ids in (low, high, missing):
                if len(ids) < 3:
                    continue
                gap = max(0.0, math.fsum((absolute[i] for i in ids)) / len(ids) - mean) * min(1.0, len(ids) / 20.0)
                local = max(local, gap)
                if len(ids) < max(8, n * 0.1):
                    small = max(small, gap)
        for a in range(min(4, len(bits))):
            for b in range(a + 1, min(4, len(bits))):
                for av, bv in ((False, False), (False, True), (True, False), (True, True)):
                    ids = [i for i in range(n) if bits[a][i] == av and bits[b][i] == bv]
                    if len(ids) >= 5:
                        signed = abs(math.fsum((residual[i] for i in ids)) / len(ids))
                        interaction = max(interaction, max(0.0, signed - bias - local) * min(1.0, len(ids) / 20.0))
        return {'INTERACTION_LIKE_ERROR': min(1.0, interaction * 3.0), 'SMALL_UNSTABLE_REGION': min(1.0, small * 4.0), 'mean_absolute_residual': mean, 'absolute_signed_bias': bias, 'local_excess': local}
