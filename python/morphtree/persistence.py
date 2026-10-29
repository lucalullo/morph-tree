"""Bounded JSON artifacts; strict graph validation, atomic writes, no pickle."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
import stat
from pathlib import Path
import tempfile
import uuid

from .config import AlgorithmGenome, HyperParams, SPEC_VERSION, VERSION, validate_task_genome
from .data import Binner, validate_names
from .tree import HistogramTree, TreeNode
from .booster import BinaryHistogramBooster
from .public import learning_fingerprint, public_metadata, validate_recipe


@dataclass(frozen=True)
class LoadLimits:
    max_bytes: int = 64 * 1024 * 1024
    max_features: int = 4096
    max_trees: int = 4096
    max_nodes: int = 262144
    max_thresholds: int = 1000000
    max_json_depth: int = 256

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError("Loader limits must be positive integers")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def regular_file_size(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Model artifact must be a regular file')
    return info.st_size


def read_regular_bytes(path, max_bytes):
    """Read one bounded regular-file snapshot, never wait for a FIFO writer.

    POSIX nonblocking open also covers a regular-path/FIFO replacement between
    stat and open. Descriptor validation precedes reading on every platform.
    Caller-supplied regular-file symlinks remain supported; checksum custody
    and typed JSON validation still apply to the exact bytes returned.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('Positive immutable snapshot allowance required')
    regular_file_size(path)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Model artifact descriptor must be a regular file')
        if info.st_size > max_bytes:
            raise ValueError('Artifact exceeds max_bytes')
        with os.fdopen(descriptor, 'rb') as handle:
            descriptor = None
            # The file ceiling is admission, not the buffer size. BufferedReader
            # may reserve its requested length even for a tiny regular file.
            # One extra byte detects growth without ceiling-sized allocation.
            raw = handle.read(info.st_size + 1)
            after = os.fstat(handle.fileno())
        if len(raw) > max_bytes or after.st_size > max_bytes:
            raise ValueError('Artifact exceeds max_bytes')
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != info.st_size:
            raise ValueError('Artifact changed while reading immutable snapshot')
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_json(path, limits=None):
    limits = limits or LoadLimits()
    return decode_json_bytes(read_regular_bytes(path, limits.max_bytes), limits)


def _admit_json_workspace(raw, limits, reserve):
    """Conservative topology forecast before stdlib parsing, constant space.

    This is resource admission, not a replacement JSON parser or an absolute
    process-memory ceiling. Syntax, duplicate keys and finite values still go
    through the strict decoder below. Charges include pairs-hook and validation
    workspace; dense containers must not hide behind a file-byte multiplier.
    """
    estimate = 4096 + len(raw) * 12
    if reserve is not None:
        reserve(estimate, 'JSON decoding workspace before parsing')
    depth = 0
    quoted = escaped = atom = False
    for position, byte in enumerate(raw):
        if quoted:
            estimate += 8
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            if depth > limits.max_json_depth:
                raise ValueError('JSON nesting depth exceeds supported limit')
            estimate += 256
            quoted = True
            atom = False
        elif byte in (123, 91):
            if depth > limits.max_json_depth:
                raise ValueError('JSON nesting depth exceeds supported limit')
            depth += 1
            estimate += 1024 if byte == 123 else 128
            atom = False
        elif byte in (125, 93):
            depth -= 1
            atom = False
        elif byte in (44, 58):
            estimate += 128
            atom = False
        elif byte in (9, 10, 13, 32):
            atom = False
        elif not atom:
            if depth > limits.max_json_depth:
                raise ValueError('JSON nesting depth exceeds supported limit')
            estimate += 128
            atom = True
        if reserve is not None and position % 4096 == 0:
            reserve(estimate, 'JSON topology workspace before parsing')
    if reserve is not None:
        reserve(estimate, 'JSON topology workspace before parsing')
    return estimate


def decode_json_bytes(raw, limits=None, *, reserve=None):
    """Validate one byte snapshot; checksum and decoder see identical bytes."""
    limits = limits or LoadLimits()
    if type(raw) is not bytes:
        raise ValueError('Immutable JSON bytes required')
    if len(raw) > limits.max_bytes:
        raise ValueError("Artifact exceeds max_bytes")
    _admit_json_workspace(raw, limits, reserve)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Malformed or excessively nested JSON artifact") from exc
    if not isinstance(value, dict):
        raise ValueError("Artifact must be a JSON object")
    # Keep iterators on the current ancestry only: a wide array must not create
    # a second width-sized list of (child,depth) tuples during validation.
    pending = [(iter((value,)), 0)]
    while pending:
        children, depth = pending[-1]
        try:
            node = next(children)
        except StopIteration:
            pending.pop()
            continue
        if depth > limits.max_json_depth:
            raise ValueError('JSON nesting depth exceeds supported limit')
        if isinstance(node, dict):
            pending.append((iter(node.values()), depth + 1))
        elif isinstance(node, list):
            pending.append((iter(node), depth + 1))
        elif isinstance(node, float) and not math.isfinite(node):
            raise ValueError("Nonfinite JSON number")
    if reserve is not None:
        reserve(0, 'JSON decoding completed')
    return value


def atomic_text(path, text):
    path = Path(path)
    # The directory must already exist. Never create arbitrary parent paths from artifact content.
    # CPython's Windows mkstemp may retry PermissionError as a name collision
    # when os.access reports writable. Do not spin through TMP_MAX candidates:
    # one unpredictable O_EXCL name, and preserve the actual filesystem error.
    temporary = path.parent / ('.morphtree-' + uuid.uuid4().hex + '.tmp')
    if os.name == 'nt':
        # A valid final path can have insufficient suffix room for a secure
        # temporary name. Extend only the owned temporary OS path; leave the
        # user destination semantics and ordinary load paths unchanged.
        temporary = os.path.abspath(temporary)
        if not temporary.startswith('\\\\?\\'):
            temporary = '\\\\?\\UNC\\' + temporary[2:] if temporary.startswith('\\\\') else '\\\\?\\' + temporary
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(temporary, flags, 0o600)
    primary = None
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            if primary is None:
                raise
            BaseException.add_note(primary, 'Owned temporary cleanup also failed: ' + type(cleanup_error).__name__)


def write_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require_keys(data, required, optional=()):
    if not isinstance(data, dict) or not set(required) <= set(data) or set(data) - set(required) - set(optional):
        raise ValueError("Unexpected or missing artifact fields")


def finite(value, name):
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a finite float") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name}")
    return value


def species_payload(genome, params, metadata=None):
    metadata = public_metadata(metadata or {})
    recipe = metadata.pop('learning_recipe', None)
    payload = {"format": "MorphTreeSpecies", "spec_version": 2 if recipe is not None else 1, "producer_version": VERSION,
            "task": "binary_classification", "capabilities": ["numeric_dense", "missing", "split_dsl_v1", "local_gates_v1"],
            "genome": genome.to_dict(), "parameters": asdict(params), "structural_fingerprint": genome.fingerprint(),
            "active_parameters": list(genome.active_parameters()), "metadata": metadata}
    if recipe is not None:
        payload.update(learning_recipe=recipe, learning_fingerprint=learning_fingerprint(genome, params, recipe))
    return payload


def decode_species(data):
    require_keys(data, ("format", "spec_version", "producer_version", "task", "capabilities", "genome", "parameters",
                        "structural_fingerprint", "active_parameters", "metadata"), ('learning_recipe', 'learning_fingerprint'))
    if data["format"] != "MorphTreeSpecies" or type(data["spec_version"]) is not int or data["spec_version"] not in (1, 2):
        raise ValueError("Unsupported species format/version; 0.2 artifacts require retraining")
    if data["task"] != "binary_classification":
        raise ValueError("Unsupported task")
    if data["capabilities"] != ["numeric_dense", "missing", "split_dsl_v1", "local_gates_v1"]:
        raise ValueError("Unsupported species capabilities")
    try:
        require_keys(data["genome"], [field.name for field in fields(AlgorithmGenome)])
        require_keys(data["parameters"], [field.name for field in fields(HyperParams)])
        genome, params = AlgorithmGenome(**data["genome"]), HyperParams(**data["parameters"])
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid species genes/parameters") from exc
    if data["structural_fingerprint"] != genome.fingerprint() or data["active_parameters"] != list(genome.active_parameters()):
        raise ValueError("Species fingerprint or active parameters mismatch")
    if not isinstance(data["metadata"], dict):
        raise ValueError("Species metadata must be an object")
    metadata = public_metadata(data['metadata'])
    metadata.pop('learning_recipe', None)
    if data['spec_version'] == 2:
        recipe = validate_recipe(data.get('learning_recipe'))
        if data.get('learning_fingerprint') != learning_fingerprint(genome, params, recipe):
            raise ValueError('Learning fingerprint mismatch')
        metadata['learning_recipe'] = recipe
    elif 'learning_recipe' in data or 'learning_fingerprint' in data:
        raise ValueError('Version 1 cannot contain version 2 learning fields')
    return genome, params, metadata


def decode_booster(data, genome, params, backend, limits=None, *, objective='binary_logloss', _aggregate_tree_count=None):
    limits = limits or LoadLimits()
    require_keys(data, ("base_score", "n_features", "binner", "steps", "trees"))
    n_features = integer(data["n_features"], "n_features", 1, limits.max_features)
    base_score = finite(data["base_score"], "base_score")
    if objective not in ('binary_logloss', 'squared_error'):
        raise ValueError('Unsupported booster objective')
    if abs(base_score) > (1e100 if objective == 'squared_error' else 100):
        raise ValueError("Implausible base score")
    bd = data["binner"]
    require_keys(bd, ("max_bins", "policy", "thresholds"))
    if integer(bd["max_bins"], "max_bins", 2, 4096) != params.max_bins or bd["policy"] != genome.binning_policy:
        raise ValueError("Binner/species mismatch")
    cuts = bd["thresholds"]
    if not isinstance(cuts, list) or len(cuts) != n_features:
        raise ValueError("Binner feature count mismatch")
    if any(not isinstance(row, list) for row in cuts) or sum(map(len, cuts)) > limits.max_thresholds:
        raise ValueError("Threshold budget exceeded")
    cuts = [[finite(v, "threshold") for v in row] for row in cuts]
    for feature,row in enumerate(cuts):
        nominal = getattr(backend, 'nominal_sizes', {}).get(feature)
        if nominal is not None:
            if row != list(range(nominal + 1)):
                raise ValueError('Nominal thresholds must match the validated TRAIN vocabulary')
        elif len(row) >= params.max_bins:
            raise ValueError("Too many thresholds for max_bins")
        if any(a >= b for a, b in zip(row, row[1:])):
            raise ValueError("Binning thresholds must be strictly increasing")
    trees, steps = data["trees"], data["steps"]
    if not isinstance(trees, list) or not isinstance(steps, list) or not 1 <= len(trees) <= limits.max_trees:
        raise ValueError("Invalid tree count")
    tree_allowance=min(params.n_estimators,limits.max_trees) if _aggregate_tree_count is None else _aggregate_tree_count
    if type(tree_allowance) is not int or not 1<=tree_allowance<=limits.max_trees:raise ValueError('Invalid aggregate tree allowance')
    if len(trees) != len(steps) or len(trees) > tree_allowance:
        raise ValueError("Tree/update count mismatch")
    steps = [finite(step, "step") for step in steps]
    for step in steps:
        if not 0 < step <= 1:
            raise ValueError("Invalid update weight")
    decoded, total_nodes = [], 0
    required_nodes = [field.name for field in fields(TreeNode)]
    for td in trees:
        require_keys(td, ("nodes",))
        nodes = td["nodes"]
        if not isinstance(nodes, list) or not nodes or len(nodes) > 2 * params.max_leaves - 1:
            raise ValueError("Invalid tree size")
        total_nodes += len(nodes)
        if total_nodes > limits.max_nodes:
            raise ValueError("Total node budget exceeded")
        for node in nodes:
            require_keys(node, required_nodes)
            if type(node["is_leaf"]) is not bool or type(node["missing_left"]) is not bool:
                raise ValueError("Node flags must be boolean")
            if abs(finite(node["value"], "leaf value")) > 1e100:
                raise ValueError("Leaf magnitude exceeds supported limit")
            integer(node["depth"], "depth", 0, params.max_depth)
            if node["is_leaf"]:
                if any(type(node[k]) is not int or node[k] != -1 for k in ("feature", "threshold_bin", "left", "right")):
                    raise ValueError("Leaf must not contain split fields")
            else:
                f = integer(node["feature"], "feature", 0, n_features - 1)
                integer(node["threshold_bin"], "threshold_bin", 0, len(cuts[f]) + 1)
                integer(node["left"], "left child", 0, len(nodes) - 1)
                integer(node["right"], "right child", 0, len(nodes) - 1)
                if node["left"] == node["right"]:
                    raise ValueError("A split must have two distinct children")
        visited, stack = set(), [(0, 0)]
        while stack:
            index, depth = stack.pop()
            if index in visited:
                raise ValueError("Cyclic tree or shared child")
            visited.add(index)
            node = nodes[index]
            if node["depth"] != depth:
                raise ValueError("Inconsistent tree depth")
            if not node["is_leaf"]:
                stack.extend(((node["left"], depth + 1), (node["right"], depth + 1)))
        if len(visited) != len(nodes):
            raise ValueError("Unreachable tree nodes")
        tree = HistogramTree(genome, params, backend)
        tree.nodes = tuple(TreeNode(**{**node, "value": finite(node["value"], "leaf value")}) for node in nodes)
        decoded.append(tree)
    booster = BinaryHistogramBooster(genome, params, backend=backend,
                                     memory_enabled=objective != 'squared_error', objective=objective)
    booster.base_score, booster.n_features_ = base_score, n_features
    booster.binner = Binner(params.max_bins, genome.binning_policy)
    booster.binner.thresholds = tuple(tuple(row) for row in cuts)
    booster.trees, booster.steps = decoded, list(steps)
    return booster


def decode_multiclass_booster(data, params, backend, n_classes, strategy, limits=None, *, genome=None):
    """Reuse the scalar graph validator with a single aggregate tree/node budget."""
    limits = limits or LoadLimits()
    if type(n_classes) is not int or n_classes<2:raise ValueError('Typed multiclass dimension required')
    require_keys(data, ('base_score', 'n_features', 'binner', 'steps', 'trees'))
    priors = data['base_score']
    if not isinstance(priors, list) or len(priors) != n_classes:
        raise ValueError('Multiclass prior dimensions disagree')
    priors = [finite(value, 'class prior score') for value in priors]
    if any(abs(value) > 100 for value in priors):
        raise ValueError('Implausible class prior score')
    trees = data['trees']
    if not isinstance(trees, list) or not trees or len(trees) % n_classes or len(trees) > limits.max_trees:
        raise ValueError('Multiclass requires complete rounds within the aggregate tree budget')
    if len(trees)//n_classes > params.n_estimators:
        raise ValueError('Multiclass round count exceeds parameters')
    # Validate once, not once per class (which would multiply the load budgets).
    genome=validate_task_genome('multiclass',AlgorithmGenome() if genome is None else genome)
    booster = decode_booster({**data, 'base_score': 0.}, genome,
        params, backend, limits, objective='squared_error', _aggregate_tree_count=len(trees))
    if any(step != params.learning_rate for step in booster.steps):
        raise ValueError('Multiclass steps must match the declared fixed learning rate')
    booster.params, booster.objective, booster.n_classes, booster.base_score = params, strategy, n_classes, priors
    return booster
