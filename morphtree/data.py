"""Input contracts, deterministic splits and compact train-only binning."""
from __future__ import annotations

from array import array
from bisect import bisect_right
import csv
import math
from numbers import Real
import random

from .config import checked_int


def get_feature_names(X):
    if not hasattr(X, "columns"):
        return None
    names = list(X.columns)
    validate_names(names, len(names))
    return names


def validate_names(names, count):
    if names is not None:
        if not isinstance(names, (list, tuple)) or len(names) != count:
            raise ValueError("Feature names must match the number of columns")
        if any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
            raise ValueError("Feature names must be unique nonempty strings")


def validate_X(X, n_features=None, expected_names=None, *, allow_empty=False):
    names = get_feature_names(X)
    if names is not None and expected_names is not None:
        if set(names) != set(expected_names):
            raise ValueError("Feature schema mismatch: missing or extra columns")
        X = X.loc[:, list(expected_names)]
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(X, "ndim") and X.ndim != 2:
        raise ValueError("X must have exactly two dimensions")
    if n_features is not None and hasattr(X, "shape") and len(X.shape) == 2 and X.shape[1] != n_features:
        raise ValueError(f"X must have exactly {n_features} features")
    if hasattr(X, "tolist"):
        X = X.tolist()
    if isinstance(X, (str, bytes)):
        raise ValueError("X must be a matrix of numeric rows")
    try:
        rows = list(X)
    except TypeError as exc:
        raise ValueError("X must be a matrix") from exc
    if not rows:
        if allow_empty and n_features is not None:
            return []
        raise ValueError("X must contain at least one row")
    width = n_features
    output = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("Each row must be a numeric sequence")
        try:
            values = list(row)
        except TypeError as exc:
            raise ValueError("X must have exactly two dimensions") from exc
        if width is None:
            width = len(values)
        if not width or len(values) != width:
            raise ValueError(f"X must have exactly {width} features in each row")
        converted = []
        for value in values:
            if value is None:
                converted.append(math.nan)
            elif isinstance(value, Real):
                try:
                    number = float(value)
                except (ValueError, OverflowError) as exc:
                    raise ValueError("Feature outside supported floating-point range") from exc
                if math.isinf(number):
                    raise ValueError("X may contain missing values but not infinity")
                converted.append(number)
            else:
                raise ValueError("X must contain numeric values, None or NaN")
        output.append(converted)
    return output


def validate_y(y, n, *, both_classes=False):
    if hasattr(y, "ndim") and y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if hasattr(y, "tolist"):
        y = y.tolist()
    if isinstance(y, (str, bytes)):
        raise ValueError("y must contain numeric 0/1 labels")
    try:
        values = list(y)
    except TypeError as exc:
        raise ValueError("y must be a one-dimensional sequence") from exc
    if len(values) != n or not values:
        raise ValueError("y must contain one label for every nonempty input row")
    if any(not isinstance(v, Real) or not math.isfinite(v) or v not in (0, 1) for v in values):
        raise ValueError("Only one-dimensional numeric 0/1 labels are supported")
    out = [int(v) for v in values]
    if both_classes and len(set(out)) != 2:
        raise ValueError("Training and selection splits must each contain both classes")
    return out


def validate_xy(X, y, n_features=None, expected_names=None, *, both_classes=True):
    X = validate_X(X, n_features, expected_names)
    return X, validate_y(y, len(X), both_classes=both_classes)


def validate_regression_y(y, n):
    """Finite scalar targets; bound magnitude to the shared tree/format range."""
    if hasattr(y, 'ndim') and y.ndim != 1:
        raise ValueError('Regression y must be one-dimensional')
    if hasattr(y, 'tolist'):
        y = y.tolist()
    if isinstance(y, (str, bytes)):
        raise ValueError('Regression y must contain real numbers')
    try:
        values = list(y)
    except TypeError as exc:
        raise ValueError('Regression y must be a one-dimensional sequence') from exc
    if len(values) != n or not values:
        raise ValueError('Regression y must contain one target for every nonempty input row')
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('Regression targets must be finite real numbers, excluding bool')
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError('Regression target outside floating-point range') from exc
        if not math.isfinite(number) or abs(number) > 1e100:
            raise ValueError('Regression targets must be finite with magnitude at most 1e100')
        output.append(number)
    return output


def validate_multiclass_y(y, n, classes=None):
    """TRAIN-derived sorted classes: homogeneous strings or signed 64-bit ints."""
    if hasattr(y, 'ndim') and y.ndim != 1:
        raise ValueError('Multiclass y must be one-dimensional')
    if hasattr(y, 'tolist'):
        y = y.tolist()
    if isinstance(y, (str, bytes)):
        raise ValueError('Multiclass y must be a label sequence')
    try:
        values = list(y)
    except TypeError as exc:
        raise ValueError('Multiclass y must be a label sequence') from exc
    if len(values) != n or not values:
        raise ValueError('One label is required for every nonempty input row')
    if not (all(type(v) is int and -(2**63) <= v < 2**63 for v in values)
            or all(type(v) is str and 0 < len(v) <= 256 for v in values)):
        raise ValueError('Classes must be homogeneous strings (1..256 chars) or signed 64-bit integers, excluding bool')
    if classes is None:
        classes = tuple(sorted(set(values)))
        if not 2 <= len(classes) <= 100:
            raise ValueError('Multiclass TRAIN requires 2..100 observed classes')
    if type(values[0]) is not type(classes[0]):
        raise ValueError('Label type does not match TRAIN classes')
    lookup = {label: i for i, label in enumerate(classes)}
    if any(value not in lookup for value in values):
        raise ValueError('Label not observed in TRAIN; class vocabulary never learns from validation/test')
    return [lookup[value] for value in values], tuple(classes)


def split_indices(y, search_fraction, guard_fraction, seed):
    rng = random.Random(seed)
    train, search, guard = [], [], []
    for label in (0, 1):
        ids = [i for i, value in enumerate(y) if value == label]
        if len(ids) < 3:
            raise ValueError("Automatic splitting requires at least 3 examples per class; provide explicit splits")
        rng.shuffle(ids)
        ns = max(1, round(len(ids) * search_fraction))
        ng = max(1, round(len(ids) * guard_fraction))
        while ns + ng >= len(ids):
            if ns >= ng and ns > 1:
                ns -= 1
            elif ng > 1:
                ng -= 1
            else:
                raise ValueError("Too few examples for three disjoint splits")
        search.extend(ids[:ns])
        guard.extend(ids[ns:ns + ng])
        train.extend(ids[ns + ng:])
    for ids in (train, search, guard):
        rng.shuffle(ids)
    return train, search, guard


def regression_split_indices(n, search_fraction, guard_fraction, seed):
    """Target-independent, reproducible partition; every row belongs to one split."""
    if n < 3:
        raise ValueError('Regression tuning requires at least 3 rows; provide explicit SEARCH and GUARD')
    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    ns = max(1, round(n * search_fraction))
    ng = max(1, round(n * guard_fraction))
    while ns + ng >= n:
        if ns >= ng and ns > 1:
            ns -= 1
        elif ng > 1:
            ng -= 1
        else:
            raise ValueError('Too few rows for three disjoint regression splits')
    return ids[ns + ng:], ids[:ns], ids[ns:ns + ng]


def subset(X, y, ids):
    return [X[i] for i in ids], [y[i] for i in ids]


def stable_midpoint(a, b):
    # Avoid overflow and keep the boundary strictly greater than a for bisect_right.
    mid = a + (b - a) / 2 if (a >= 0) == (b >= 0) else a / 2 + b / 2
    return b if mid <= a else mid


class Binner:
    """Bin zero is missing; observed bins are 1..max_bins."""
    def __init__(self, max_bins=31, policy="quantile"):
        checked_int("max_bins", max_bins, 2, 4096)
        if policy not in ("quantile", "uniform"):
            raise ValueError("Unknown binning policy")
        self.max_bins, self.policy = max_bins, policy
        self.thresholds = None

    def fit(self, X):
        cuts = []
        for j in range(len(X[0])):
            values = sorted(row[j] for row in X if not math.isnan(row[j]))
            unique = sorted(set(values))
            if len(unique) <= 1:
                thresholds = []
            elif len(unique) <= self.max_bins:
                thresholds = [stable_midpoint(a, b) for a, b in zip(unique, unique[1:])]
            elif self.policy == "uniform":
                low, high = unique[0], unique[-1]
                thresholds = [low * (1 - k / self.max_bins) + high * (k / self.max_bins) for k in range(1, self.max_bins)]
            else:
                thresholds = [values[min(len(values) - 1, (k * len(values)) // self.max_bins)] for k in range(1, self.max_bins)]
            cuts.append(tuple(sorted({v for v in thresholds if unique[0] < v <= unique[-1]})) if thresholds else ())
        self.thresholds = tuple(cuts)
        return self

    def transform(self, X):
        if self.thresholds is None:
            raise RuntimeError("Binner is not fitted")
        kind = "B" if self.max_bins <= 255 else "H"
        return [array(kind, (0 if math.isnan(row[j]) else bisect_right(cuts, row[j]) + 1 for row in X))
                for j, cuts in enumerate(self.thresholds)]

    def to_dict(self):
        return {"max_bins": self.max_bins, "policy": self.policy, "thresholds": [list(c) for c in self.thresholds]}


def read_csv(path, target=None, expected_names=None):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            names = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
        validate_names(names, len(names))
        if target is not None and target not in names:
            raise ValueError(f"Target column {target!r} not found")
        features = [c for c in names if c != target]
        if expected_names is not None:
            if set(features) != set(expected_names):
                raise ValueError("CSV feature schema mismatch")
            features = list(expected_names)
        indices = [names.index(c) for c in features]
        X, y = [], []
        for line, row in enumerate(reader, 2):
            if len(row) != len(names):
                raise ValueError(f"Malformed CSV row {line}")
            try:
                X.append([float(row[j]) if row[j].strip().lower() not in ("", "nan", "none", "null") else None for j in indices])
                if target is not None:
                    y.append(float(row[names.index(target)]))
            except ValueError as exc:
                raise ValueError(f"Nonnumeric CSV value at row {line}") from exc
    return X, y if target is not None else None, features
