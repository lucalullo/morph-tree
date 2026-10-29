"""Reference backend: standard library only, column-major compact bins."""
from __future__ import annotations

from array import array
import math
from .metrics import sigmoid, softmax


class PythonBackend:
    name = "python-stdlib"

    def bin(self, binner, X):
        return binner.transform(X)

    def gradients(self, raw, y):
        probabilities = [sigmoid(v) for v in raw]
        gradient = array("d", (p - label for p, label in zip(probabilities, y)))
        hessian = array("d", (max(p * (1 - p), 1e-12) for p in probabilities))
        return probabilities, gradient, hessian

    def probabilities(self, raw):
        return [sigmoid(v) for v in raw]

    def regression_gradients(self, raw, y):
        # L = 0.5 * (prediction - target)^2: g = prediction - target, h = 1.
        return raw, array('d', (v - target for v, target in zip(raw, y))), array('d', [1.0]) * len(y)

    def multiclass_raw(self, prior, n):
        return [list(prior) for _ in range(n)]

    def multiclass_probabilities(self, raw, strategy, *, normalize=True):
        if strategy == 'softmax':
            return [softmax(row) for row in raw]
        probabilities = [[sigmoid(float(v)) for v in row] for row in raw]
        if normalize:
            for row in probabilities:
                total = math.fsum(row)
                for j in range(len(row)):
                    row[j] /= total
        return probabilities

    def multiclass_gradient_column(self, probabilities, y, column, strategy):
        # D=2*diag(p*(1-p)) majorizes diag(p)-p*p^T. It is NOT the exact Hessian.
        factor = 2. if strategy == 'softmax' else 1.
        return (array('d', (row[column] - (label == column) for row, label in zip(probabilities, y))),
                array('d', (max(factor * row[column] * (1-row[column]), 1e-12) for row in probabilities)))

    def multiclass_update(self, raw, column, step, prediction):
        for row, value in zip(raw, prediction):
            row[column] += step * float(value)

    def center_multiclass(self, raw):
        for row in raw:
            mean = math.fsum(row) / len(row)
            for j in range(len(row)):
                row[j] -= mean

    def node_statistics(self, Xb, ids, g, h, u, features, need_missing, need_uncertainty):
        count = max(len(ids), 1)
        missing = sum(any(Xb[f][i] == 0 for f in features) for i in ids) / count if need_missing else 0.0
        uncertainty = math.fsum(u[i] for i in ids) / count if need_uncertainty else 0.0
        return math.fsum(g[i] for i in ids), math.fsum(h[i] for i in ids), uncertainty, missing

    def histogram(self, column, indices, g, h, r, u, size):
        result = [array("d", [0.0]) * size for _ in range(5)]
        counts, gs, hs, rs, us = result
        for i in indices:
            b = column[i]
            counts[b] += 1
            gs[b] += g[i]
            hs[b] += h[i]
            rs[b] += r[i]
            us[b] += u[i]
        return result

    def subtract(self, parent, child):
        result = [[a - b for a, b in zip(pa, ch)] for pa, ch in zip(parent, child)]
        result[0] = [max(0.0, c) for c in result[0]]
        result[2] = [max(0.0, h) for h in result[2]]
        return result

    def cumulative(self, histogram):
        result = []
        for row in histogram:
            running, sums = 0.0, []
            for value in row[1:]:
                running += value
                sums.append(running)
            result.append(sums)
        return result

    def partition(self, column, ids, cut, missing_left):
        left, right = [], []
        for i in ids:
            b = column[i]
            goes_left = missing_left if b == 0 else b <= cut
            (left if goes_left else right).append(i)
        return left, right

    def predict_tree(self, nodes, Xb):
        n = len(Xb[0])
        out = [0.0] * n
        for row in range(n):
            node_id = 0
            for _ in range(len(nodes) + 1):
                node = nodes[node_id]
                if node.is_leaf:
                    out[row] = node.value
                    break
                b = Xb[node.feature][row]
                goes_left = node.missing_left if b == 0 else b <= node.threshold_bin
                node_id = node.left if goes_left else node.right
            else:
                raise ValueError("Invalid cyclic tree")
        return out

    def add_update(self, raw, step, prediction):
        return [r + step * v for r, v in zip(raw, prediction)]

    def fast_split(self, hist, total, depth, genome, params):
        if genome.split_program != ("OBJECTIVE_GAIN",):
            return NotImplemented
        # Canonical scalar arithmetic, specialized to the three consumed rows.
        # Hoist regularization; do not allocate left/right lists for every cut.
        counts, gradients, hessians = ([float(v) for v in row] for row in hist[:3])
        n, G, H = total[:3]
        factor = params.depth_reg_strength if genome.regularization_policy == "depth_scaled" else 0.
        child_l2 = params.l2 * (1 + factor * (depth + 1))
        parent_l2 = params.l2 * (1 + factor * depth)
        l1 = params.l1
        thresholded = math.copysign(max(abs(G)-l1, 0.), G)
        parent_score = thresholded * thresholded / max(H+parent_l2, 1e-12)
        directions = (True, False) if genome.missing_policy == "learned" else (genome.missing_policy == "left",)
        if len(directions) == 2 and not any(row[0] for row in hist):
            directions = directions[:1]
        mn, mg, mh = counts[0], gradients[0], hessians[0]
        cn = cg = ch = 0.
        best = None
        best_rounded = None
        support, min_hessian, min_gain = params.min_samples_leaf, params.min_hessian, params.min_gain
        for cut in range(len(counts)):
            if cut:
                cn += counts[cut]
                cg += gradients[cut]
                ch += hessians[cut]
            for missing_left in directions:
                ln = cn + (mn if missing_left else 0.)
                lg = cg + (mg if missing_left else 0.)
                lh = ch + (mh if missing_left else 0.)
                rn, rg, rh = n-ln, G-lg, H-lh
                if min(ln, rn) < support or min(lh, rh) < min_hessian:
                    continue
                lt = math.copysign(max(abs(lg)-l1, 0.), lg)
                rt = math.copysign(max(abs(rg)-l1, 0.), rg)
                left_score = lt * lt / max(lh+child_l2, 1e-12)
                right_score = rt * rt / max(rh+child_l2, 1e-12)
                gain = 0.5 * (left_score + right_score - parent_score)
                if math.isfinite(gain) and gain > min_gain:
                    rounded = round(gain, 12)
                    if best is None or rounded > best_rounded:
                        best, best_rounded = (gain, cut, missing_left), rounded
        return best
