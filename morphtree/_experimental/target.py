import math,random
from .preprocessing import _m26_scalar
def detect_target(y, task):
    if hasattr(y, 'tolist'):
        y = y.tolist()
    if not isinstance(y, (list, tuple)) or len(y) < 4:
        raise ValueError('At least four scalar target values required')
    values = [_m26_scalar(v) for v in y]
    if any((v is None for v in values)):
        raise ValueError('Missing target is unsupported')
    strings = all((isinstance(v, str) for v in values))
    numeric = all((type(v) is float for v in values))
    if not (strings or numeric):
        raise ValueError('Target must be consistently string or numeric')
    classes = sorted(set(values))
    if task == 'auto':
        if strings:
            task = 'binary' if len(classes) == 2 else 'multiclass'
        elif classes == [0.0, 1.0]:
            task = 'binary'
        elif len(classes) > 2 and any((not v.is_integer() for v in values)):
            task = 'regression'
        else:
            raise ValueError('Ambiguous numeric target: specify task="regression", "binary" or "multiclass"')
    if task == 'regression':
        if not numeric:
            raise ValueError('Regression requires finite numeric targets')
        return (task, values, [])
    if task not in ('binary', 'multiclass') or not 2 <= len(classes) <= 64 or (task == 'binary' and len(classes) != 2):
        raise ValueError('Classification requires two (binary) or 2..64 (multiclass) classes')
    return (task, [classes.index(v) for v in values], classes)

def _split(y, task, seed):
    rng = random.Random(seed)
    held = []
    groups = [list(range(len(y)))] if task == 'regression' else [[i for i, v in enumerate(y) if v == c] for c in sorted(set(y))]
    for ids in groups:
        if len(ids) < 2:
            raise ValueError('Each class needs at least two rows for disjoint TRAIN/SEARCH')
        rng.shuffle(ids)
        held.extend(ids[:max(1, len(ids) // 4)])
    train = sorted(set(range(len(y))) - set(held))
    return (train, sorted(held))

class _NamedRows:
    """Local duck-typed carrier; avoids optional pandas or implicit column loss."""

    def __init__(self, rows, names):
        self.rows = rows
        if names is not None:
            self.columns = names

    def to_numpy(self):
        return self.rows
