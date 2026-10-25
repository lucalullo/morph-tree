"""NumPy kernels only. Training still uses the native MorphTree tree builder."""
from __future__ import annotations

import numpy as np
import math
from .backend import PythonBackend
from .metrics import sigmoid

_FLOAT64_EPS = float(np.finfo(np.float64).eps)


class NumpyBackend(PythonBackend):
    name = "numpy"

    def __init__(self):
        self.split_calls = 0
        self.canonical_split_fallbacks = 0

    def bin(self, binner, X):
        matrix = np.asarray(X, dtype=np.float64)
        out = np.empty((len(binner.thresholds), len(X)), dtype=np.uint8 if binner.max_bins <= 255 else np.uint16)
        for j, cuts in enumerate(binner.thresholds):
            values = matrix[:, j]
            out[j] = np.where(np.isnan(values), 0, np.searchsorted(cuts, values, side="right") + 1)
        return out

    def gradients(self, raw, y):
        values = np.asarray(raw, dtype=np.float64)
        probabilities = np.fromiter((sigmoid(float(v)) for v in values), dtype=np.float64, count=len(values))
        return probabilities.tolist(), probabilities - np.asarray(y), np.maximum(probabilities * (1 - probabilities), 1e-12)

    def probabilities(self, raw):
        return [sigmoid(float(v)) for v in raw]

    def regression_gradients(self, raw, y):
        values = np.asarray(raw, dtype=np.float64)
        return values.tolist(), values - np.asarray(y, dtype=np.float64), np.ones(len(y), dtype=np.float64)

    def multiclass_raw(self, prior, n):
        return np.tile(np.asarray(prior, dtype=np.float64), (n, 1))

    def multiclass_gradient_column(self, probabilities, y, column, strategy):
        values = np.fromiter((row[column] for row in probabilities), dtype=np.float64, count=len(y))
        factor = 2. if strategy == 'softmax' else 1.
        return values - (np.asarray(y) == column), np.maximum(factor*values*(1-values), 1e-12)

    def multiclass_update(self, raw, column, step, prediction):
        raw[:, column] += step * np.asarray(prediction)

    def node_statistics(self, Xb, ids, g, h, u, features, need_missing, need_uncertainty):
        indices = np.asarray(ids, dtype=np.intp)
        missing = float(np.mean(np.any(Xb[np.ix_(features, indices)] == 0, axis=0))) if need_missing else 0.0
        # Gather in C, then keep the reference fsum on the same ordered floats.
        uncertainty = math.fsum(np.asarray(u)[indices].tolist()) / max(len(ids), 1) if need_uncertainty else 0.0
        return math.fsum(np.asarray(g)[indices].tolist()), math.fsum(np.asarray(h)[indices].tolist()), uncertainty, missing

    def histogram(self, column, indices, g, h, r, u, size):
        ids = np.asarray(indices, dtype=np.intp)
        bins = column[ids]
        return np.vstack((np.bincount(bins, minlength=size),
                          *(np.bincount(bins, weights=np.asarray(values)[ids], minlength=size) for values in (g, h, r, u))))

    def subtract(self, parent, child):
        out = parent - child
        out[0] = np.maximum(out[0], 0)
        out[2] = np.maximum(out[2], 0)
        return out

    def cumulative(self, histogram):
        return np.cumsum(histogram[:, 1:], axis=1)

    def partition(self, column, ids, cut, missing_left):
        indices = np.asarray(ids, dtype=np.intp)
        bins = column[indices]
        mask = np.where(bins == 0, missing_left, bins <= cut)
        return indices[mask].tolist(), indices[~mask].tolist()

    def add_update(self, raw, step, prediction):
        return np.asarray(raw, dtype=np.float64) + step * prediction

    def predict_tree(self, nodes, Xb):
        # Tiny batches spend more time allocating vector queues than traversing.
        # The reference walk preserves missing routes and cycle bounds exactly.
        if Xb.shape[1] <= 16:
            return np.asarray(PythonBackend.predict_tree(self, nodes, Xb), dtype=np.float64)
        out = np.empty(Xb.shape[1], dtype=np.float64)
        queue = [(0, np.arange(Xb.shape[1]))]
        visits = 0
        while queue:
            index, rows = queue.pop()
            visits += 1
            if visits > len(nodes):
                raise ValueError("Invalid cyclic tree")
            if not len(rows):
                continue
            node = nodes[index]
            if node.is_leaf:
                out[rows] = node.value
                continue
            bins = Xb[node.feature, rows]
            mask = np.where(bins == 0, node.missing_left, bins <= node.threshold_bin)
            if mask.any():
                queue.append((node.left, rows[mask]))
            if (~mask).any():
                queue.append((node.right, rows[~mask]))
        return out

    def fast_split(self, hist, total, depth, genome, params):
        self.split_calls += 1
        directions = np.asarray((True, False) if genome.missing_policy == "learned" else (genome.missing_policy == "left",))
        if len(directions) == 2 and not hist[:, 0].any():
            # Both routes are exactly identical; retain the core's first tie.
            directions = directions[:1]
        cumulative = np.column_stack((np.zeros(5), np.cumsum(hist[:, 1:], axis=1)))
        left = cumulative[:, :, None] + hist[:, 0, None, None] * directions[None, None, :]
        right = np.asarray(total)[:, None, None] - left
        ln, lg, lh, lr, lu = left
        rn, rg, rh, rr, ru = right
        n, G, H, R, U = total
        valid = (ln >= params.min_samples_leaf) & (rn >= params.min_samples_leaf) & (lh >= params.min_hessian) & (rh >= params.min_hessian)
        if not valid.any():
            return None
        factor = params.depth_reg_strength if genome.regularization_policy == "depth_scaled" else 0.0
        l2_child = params.l2 * (1 + factor * (depth + 1))
        l2_parent = params.l2 * (1 + factor * depth)
        score_left = np.maximum(np.abs(lg) - params.l1, 0) ** 2 / np.maximum(lh + l2_child, 1e-12)
        score_right = np.maximum(np.abs(rg) - params.l1, 0) ** 2 / np.maximum(rh + l2_child, 1e-12)
        parent = max(abs(G) - params.l1, 0) ** 2 / max(H + l2_parent, 1e-12)
        # Programs usually consume one signal; do not allocate unused arrays.
        program = genome.split_program
        signals = {"OBJECTIVE_GAIN": 0.5 * (score_left + score_right - parent)}
        if 'RESIDUAL_VARIANCE_GAIN' in program or 'UNCERTAINTY_GAIN' in program:
            safe_ln, safe_rn = np.maximum(ln, 1), np.maximum(rn, 1)
        if 'RESIDUAL_VARIANCE_GAIN' in program:
            signals['RESIDUAL_VARIANCE_GAIN'] = np.maximum(0, lr * lr / safe_ln + rr * rr / safe_rn - R * R / n)
        if 'BALANCE' in program:
            signals['BALANCE'] = 2 * np.minimum(ln, rn) / n
        if 'HESSIAN_BALANCE' in program:
            signals['HESSIAN_BALANCE'] = 2 * np.minimum(lh, rh) / max(H, 1e-12)
        if 'UNCERTAINTY_GAIN' in program:
            signals['UNCERTAINTY_GAIN'] = (lu / safe_ln - ru / safe_rn) ** 2 * ln * rn / n
        stack = []
        for token in genome.split_program:
            if token in signals:
                stack.append(signals[token])
            elif token.startswith("MUL_SLOT:"):
                stack[-1] = stack[-1] * getattr(params, f"signal_slot_{token[-1]}")
            else:
                b, a = stack.pop(), stack.pop()
                if token == "ADD":
                    stack.append(a + b)
                elif token == "MUL":
                    stack.append(a * b)
                elif token == "MIN":
                    stack.append(np.minimum(a, b))
                else:
                    stack.append(np.maximum(a, b))
        finite_valid = valid & np.isfinite(stack[0])
        if not finite_valid.any():
            return None
        maximum = float(np.max(stack[0][finite_valid]))
        # Conservative error envelope: near admission or a rounding boundary,
        # the core evaluates the SAME histogram with canonical scalar operations.
        # This does not change min_gain or the probability tolerance.
        scale = max(1., abs(parent), float(score_left.max()), float(score_right.max()), abs(maximum))
        error = 64 * _FLOAT64_EPS * scale
        contenders = stack[0][finite_valid & (stack[0] >= maximum - 1e-12 - error)]
        rounding_ambiguous = len(contenders) > 1 and float(contenders.min()) != maximum and np.any(
            np.abs((contenders * 1e12) % 1 - .5) * 1e-12 <= error)
        if abs(maximum - params.min_gain) <= error or rounding_ambiguous:
            self.canonical_split_fallbacks += 1
            return NotImplemented
        gains = np.where(finite_valid & (stack[0] > params.min_gain), stack[0], -np.inf)
        # Same 12-decimal score ordering and first-in-order tie rule as the core.
        flat = int(np.argmax(np.round(gains, 12)))
        cut, direction_index = np.unravel_index(flat, gains.shape)
        gain = float(gains[cut, direction_index])
        if gain <= params.min_gain:
            return None
        return gain, int(cut), bool(directions[direction_index])
