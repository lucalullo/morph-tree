"""Histogram GBDT tree, explicit missing splits and histogram subtraction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import heapq
import math

from .backend import PythonBackend


def run_program(program, signals, slots):
    """Small bounded stack interpreter. No eval, exec or code compilation."""
    stack = []
    for token in program:
        if token in signals:
            stack.append(signals[token])
        elif token.startswith("MUL_SLOT:"):
            stack[-1] *= slots[int(token[-1])]
        else:
            right, left = stack.pop(), stack.pop()
            if token == "ADD":
                stack.append(left + right)
            elif token == "MUL":
                stack.append(left * right)
            elif token == "MIN":
                stack.append(min(left, right))
            elif token == "MAX":
                stack.append(max(left, right))
            else:
                raise ValueError("Unknown program instruction")
    return stack[0]


def soft_threshold(g, l1):
    return math.copysign(max(abs(g) - l1, 0.0), g)


@dataclass(frozen=True)
class TreeNode:
    is_leaf: bool = True
    value: float = 0.0
    feature: int = -1
    threshold_bin: int = -1
    missing_left: bool = True
    left: int = -1
    right: int = -1
    depth: int = 0


class HistogramTree:
    def __init__(self, genome, params, backend=None):
        self.genome, self.params = genome, params
        self.backend = backend or PythonBackend()
        self.nodes = ()
        self.root_count = 0
        self.histogram_builds = 0
        self.histogram_subtractions = 0

    def effective_l2(self, depth):
        factor = 1 + self.params.depth_reg_strength * depth if self.genome.regularization_policy == "depth_scaled" else 1
        return self.params.l2 * factor

    def score(self, g, h, depth):
        thresholded = soft_threshold(g, self.params.l1)
        return thresholded * thresholded / max(h + self.effective_l2(depth), 1e-12)

    def leaf_value(self, g, h, count, depth, uncertainty, missing_ratio):
        value = -soft_threshold(g, self.params.l1) / max(h + self.effective_l2(depth), 1e-12)
        if self.genome.leaf_policy == "smoothed":
            value *= count / (count + self.params.path_smooth)
        elif self.genome.leaf_policy == "clipped":
            value = max(-self.params.max_delta_step, min(self.params.max_delta_step, value))
        elif self.genome.leaf_policy == "uncertainty":
            value /= 1 + self.params.gate_strength * uncertainty
        gate = (self.genome.local_policy == "low_support" and count < max(8, self.root_count / 16))
        gate = gate or (self.genome.local_policy == "missing_heavy" and missing_ratio >= 0.5)
        if gate:
            value /= 1 + self.params.gate_strength
        return value

    def _histograms(self, Xb, ids, g, h, r, u, features, sizes):
        self.histogram_builds += len(features)
        return {f: self.backend.histogram(Xb[f], ids, g, h, r, u, sizes[f]) for f in features}

    def _best_split(self, histograms, depth, features):
        best = None
        if depth >= self.params.max_depth:
            return None
        directions = (True, False) if self.genome.missing_policy == "learned" else (self.genome.missing_policy == "left",)
        for feature in features:
            hist = histograms[feature]
            total = [math.fsum(row) for row in hist]
            n, G, H, R, U = total
            if n < 2 * self.params.min_samples_leaf:
                continue
            fast = self.backend.fast_split(hist, total, depth, self.genome, self.params)
            if fast is NotImplemented and self.genome.split_program == ("OBJECTIVE_GAIN",):
                # NumPy's ambiguous-gain fallback keeps canonical arithmetic.
                fast = PythonBackend().fast_split(hist, total, depth, self.genome, self.params)
            if fast is not NotImplemented:
                if fast is not None and (best is None or round(fast[0], 12) > round(best[0], 12)):
                    best = (fast[0], feature, fast[1], fast[2])
                continue
            # Python float arithmetic is the admission/tie reference, including
            # the narrowly requested NumPy fallback for ambiguous gains.
            hist = [[float(v) for v in row] for row in hist]
            # Without missing observations both routes have identical statistics.
            # Preserve the historical first tie (missing_left=True), as NumPy does.
            feature_directions = directions[:1] if len(directions) == 2 and not any(row[0] for row in hist) else directions
            cumulative = PythonBackend().cumulative(hist)
            parent_score = self.score(G, H, depth)
            # Cut 0 isolates missing values. Final cut isolates observed from missing.
            for cut in range(len(hist[0])):
                observed = [row[cut - 1] if cut else 0.0 for row in cumulative]
                for missing_left in feature_directions:
                    left = [v + (hist[k][0] if missing_left else 0.0) for k, v in enumerate(observed)]
                    ln, lg, lh, lr, lu = left
                    rn, rg, rh, rr, ru = [v - vleft for v, vleft in zip(total, left)]
                    if min(ln, rn) < self.params.min_samples_leaf or min(lh, rh) < self.params.min_hessian:
                        continue
                    objective = 0.5 * (self.score(lg, lh, depth + 1) + self.score(rg, rh, depth + 1) - parent_score)
                    if self.genome.split_program == ("OBJECTIVE_GAIN",):
                        gain = objective
                    else:
                        signals = {"OBJECTIVE_GAIN": objective,
                                   "RESIDUAL_VARIANCE_GAIN": max(0.0, lr * lr / ln + rr * rr / rn - R * R / n),
                                   "BALANCE": 2 * min(ln, rn) / n,
                                   "HESSIAN_BALANCE": 2 * min(lh, rh) / max(H, 1e-12),
                                   "UNCERTAINTY_GAIN": (lu / ln - ru / rn) ** 2 * ln * rn / n}
                        gain = run_program(self.genome.split_program, signals, (self.params.signal_slot_0, self.params.signal_slot_1))
                    if math.isfinite(gain) and gain > self.params.min_gain and (best is None or round(gain, 12) > round(best[0], 12)):
                        best = (gain, feature, cut, missing_left)
        return best

    def _lookahead_split(self, Xb, ids, g, h, r, u, features, sizes, depth):
        """Bounded two-level interaction probe; used only when greedy splitting stalls."""
        if depth + 1 >= self.params.max_depth or len(ids) < 4 * self.params.min_samples_leaf:
            return None
        best = None
        directions = (True, False) if self.genome.missing_policy == "learned" else (self.genome.missing_policy == "left",)
        for feature in features[:4]:
            observed = sorted({int(Xb[feature][i]) for i in ids if Xb[feature][i] > 0})
            if not observed:
                continue
            cuts = sorted({observed[min(len(observed) - 1, k * len(observed) // 4)] for k in range(4)})
            for cut in cuts:
                for missing_left in directions:
                    left, right = self.backend.partition(Xb[feature], ids, cut, missing_left)
                    if min(len(left), len(right)) < self.params.min_samples_leaf:
                        continue
                    future_gain = 0.0
                    for child in (left, right):
                        if len(child) < 2 * self.params.min_samples_leaf:
                            continue
                        hist = self._histograms(Xb, child, g, h, r, u, features, sizes)
                        split = self._best_split(hist, depth + 1, features)
                        if split is not None:
                            future_gain += split[0]
                    G = math.fsum(g[i] for i in ids)
                    H = math.fsum(h[i] for i in ids)
                    lg = math.fsum(g[i] for i in left)
                    lh = math.fsum(h[i] for i in left)
                    rh = H - lh
                    if min(lh, rh) < self.params.min_hessian:
                        continue
                    immediate = 0.5 * (self.score(lg, lh, depth + 1) + self.score(G - lg, rh, depth + 1) - self.score(G, H, depth))
                    gain = immediate + future_gain
                    if gain > self.params.min_gain and (best is None or round(gain, 12) > round(best[0], 12)):
                        best = (gain, feature, cut, missing_left)
        return best

    def fit(self, Xb, g, h, r, u, rng, *, indices=None, sizes=None):
        n_features = len(Xb)
        features = sorted(rng.sample(range(n_features), max(1, math.ceil(n_features * self.params.feature_fraction))))
        ids = list(range(len(g))) if indices is None else list(indices)
        self.root_count = len(ids)
        sizes = sizes or [max(column, default=0) + 1 for column in Xb]
        hist = self._histograms(Xb, ids, g, h, r, u, features, sizes)

        def make_node(node_ids, depth):
            count = len(node_ids)
            G, H, uncertainty, missing = self.backend.node_statistics(
                Xb, node_ids, g, h, u, features, self.genome.local_policy == "missing_heavy", self.genome.leaf_policy == "uncertainty")
            val = self.leaf_value(G, H, count, depth, uncertainty, missing)
            return TreeNode(value=val, depth=depth)

        nodes = [make_node(ids, 0)]
        queue, serial = [], 0

        def enqueue(node_id, node_ids, node_hist):
            nonlocal serial
            if len(nodes) >= 2 * self.params.max_leaves - 1:
                return
            split = self._best_split(node_hist, nodes[node_id].depth, features)
            if split is None and self.genome.growth_policy == "lookahead":
                split = self._lookahead_split(Xb, node_ids, g, h, r, u, features, sizes, nodes[node_id].depth)
            if split is None:
                return
            priority = nodes[node_id].depth if self.genome.growth_policy == "depthwise" else -round(split[0], 12)
            heapq.heappush(queue, (priority, serial, node_id, node_ids, node_hist, split))
            serial += 1

        enqueue(0, ids, hist)
        leaves = 1
        while queue and leaves < self.params.max_leaves:
            _, _, node_id, node_ids, parent_hist, split = heapq.heappop(queue)
            _, feature, cut, missing_left = split
            left_ids, right_ids = self.backend.partition(Xb[feature], node_ids, cut, missing_left)
            left_id, right_id = len(nodes), len(nodes) + 1
            depth = nodes[node_id].depth + 1
            nodes.extend((make_node(left_ids, depth), make_node(right_ids, depth)))
            nodes[node_id] = replace(nodes[node_id], is_leaf=False, feature=feature, threshold_bin=cut,
                                     missing_left=bool(missing_left), left=left_id, right=right_id)
            leaves += 1
            if leaves >= self.params.max_leaves:
                break
            small_ids = left_ids if len(left_ids) <= len(right_ids) else right_ids
            small_hist = self._histograms(Xb, small_ids, g, h, r, u, features, sizes)
            large_hist = {f: self.backend.subtract(parent_hist[f], small_hist[f]) for f in features}
            self.histogram_subtractions += len(features)
            left_hist, right_hist = (small_hist, large_hist) if small_ids is left_ids else (large_hist, small_hist)
            enqueue(left_id, left_ids, left_hist)
            enqueue(right_id, right_ids, right_hist)
        self.nodes = tuple(nodes)
        return self

    def predict_binned(self, Xb):
        return self.backend.predict_tree(self.nodes, Xb)

    def to_dict(self):
        return {"nodes": [asdict(node) for node in self.nodes]}
