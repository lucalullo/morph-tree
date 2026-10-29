"""Binary metrics with explicit shape and finiteness checks."""
from __future__ import annotations

import math
from numbers import Real
from .data import validate_y, validate_regression_y


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 745.0)))
    z = math.exp(max(value, -745.0))
    return z / (1.0 + z)


def checked_probabilities(y, p):
    if hasattr(p, "ndim") and p.ndim != 1:
        raise ValueError("Use one probability per sample, not a two-column matrix")
    values = list(p)
    labels = validate_y(y, len(values))
    if any(not isinstance(v, Real) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return labels, values


def point_losses(y, p):
    labels, values = checked_probabilities(y, p)
    return [-math.log(max(1e-15, min(1 - 1e-15, v if label else 1 - v))) for label, v in zip(labels, values)]


def logloss(y, p):
    losses = point_losses(y, p)
    return math.fsum(losses) / len(losses)


def accuracy(y, p):
    labels, values = checked_probabilities(y, p)
    return sum((v >= 0.5) == label for label, v in zip(labels, values)) / len(labels)


def binary_metrics(y, p):
    labels, values = checked_probabilities(y, p)
    counts = [labels.count(c) for c in (0, 1)]
    recalls = [sum((v >= 0.5) == c and label == c for label, v in zip(labels, values)) / counts[c]
               for c in (0, 1) if counts[c]]
    auc = None
    if all(counts):
        order = sorted(zip(values, labels))
        rank_sum, i = 0.0, 0
        while i < len(order):
            j = i + 1
            while j < len(order) and order[j][0] == order[i][0]:
                j += 1
            rank_sum += (i + 1 + j) / 2 * sum(label for _, label in order[i:j])
            i = j
        auc = (rank_sum - counts[1] * (counts[1] + 1) / 2) / (counts[0] * counts[1])
    return {"logloss": logloss(labels, values), "accuracy": accuracy(labels, values),
            "balanced_accuracy": sum(recalls) / len(recalls), "roc_auc": auc,
            "brier_score": math.fsum((v - label) ** 2 for label, v in zip(labels, values)) / len(labels)}


def paired_improvement(y, old_p, new_p, z=0.0):
    old, new = point_losses(y, old_p), point_losses(y, new_p)
    differences = [a - b for a, b in zip(old, new)]
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((d - mean) ** 2 for d in differences) / max(1, len(differences) - 1)
    se = math.sqrt(variance / len(differences))
    return mean, mean - z * se, se


def _regression_pairs(y, prediction):
    if isinstance(prediction, (str, bytes)) or (hasattr(prediction, 'ndim') and prediction.ndim != 1):
        raise ValueError('Use one continuous prediction per row')
    try:
        values = list(prediction)
    except TypeError as exc:
        raise ValueError('Predictions must be a one-dimensional sequence') from exc
    labels = validate_regression_y(y, len(values))
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('Predictions must be finite real numbers')
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError('Prediction outside floating-point range') from exc
        if not math.isfinite(number):
            raise ValueError('Predictions must be finite')
        normalized.append(number)
    residuals = [a - b for a, b in zip(labels, normalized)]
    if any(not math.isfinite(v) for v in residuals):
        raise ArithmeticError('Regression residual outside floating-point range')
    return labels, residuals


def _root_mean_square(values):
    scale = max(map(abs, values), default=0.0)
    return scale * math.sqrt(math.fsum((v / scale) ** 2 for v in values) / len(values)) if scale else 0.0


def rmse(y, prediction):
    """Scaled sum of squares avoids squaring very large/tiny residuals directly."""
    return _root_mean_square(_regression_pairs(y, prediction)[1])


def mae(y, prediction):
    residuals = _regression_pairs(y, prediction)[1]
    return math.fsum(abs(v) / len(residuals) for v in residuals)


def r2_score(y, prediction):
    labels, residuals = _regression_pairs(y, prediction)
    error = _root_mean_square(residuals)
    if all(v == labels[0] for v in labels):
        return 1.0 if error == 0 else 0.0
    mean = math.fsum(v / len(labels) for v in labels)
    spread = _root_mean_square([v - mean for v in labels])
    # Constant targets, including a single row: explicitly finite convention.
    if spread == 0:
        return 1.0 if error == 0 else 0.0
    ratio = error / spread
    result = 1 - ratio * ratio
    if not math.isfinite(result):
        raise ArithmeticError('R2 outside floating-point range')
    return result


def regression_metrics(y, prediction):
    return {'rmse': rmse(y, prediction), 'mae': mae(y, prediction), 'r2': r2_score(y, prediction)}
def softmax(scores):
    """Stable categorical probabilities; callers supply finite nonempty logits."""
    maximum = max(scores)
    weights = [math.exp(float(value) - maximum) for value in scores]
    total = math.fsum(weights)
    return [value / total for value in weights]


def multiclass_metrics(y, probabilities):
    """Encoded-label logloss, accuracy and macro F1 (absent classes contribute 0)."""
    if not y or len(y) != len(probabilities) or not probabilities or len(probabilities[0]) < 2:
        raise ValueError('Multiclass metrics require nonempty aligned labels and probabilities')
    k = len(probabilities[0])
    tp, actual, predicted, losses = [0]*k, [0]*k, [0]*k, []
    for label, row in zip(y, probabilities):
        if type(label) is not int or not 0 <= label < k or len(row) != k:
            raise ValueError('Invalid multiclass label/probability dimensions')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in row) or abs(math.fsum(row) - 1) > 1e-10:
            raise ValueError('Probabilities must be finite and sum to one')
        winner = max(range(k), key=row.__getitem__)
        actual[label] += 1
        predicted[winner] += 1
        tp[label] += winner == label
        losses.append(-math.log(max(float(row[label]), 1e-15)))
    return {'logloss': math.fsum(losses)/len(y), 'accuracy': sum(tp)/len(y),
            'macro_f1': math.fsum(2*tp[i]/(actual[i]+predicted[i]) if actual[i]+predicted[i] else 0. for i in range(k))/k}

