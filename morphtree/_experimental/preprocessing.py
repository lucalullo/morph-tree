"""Bounded TRAIN-local schema shared by the public runtime and standalone export.

String categoricals use sorted one-hot columns plus separate missing/unseen
channels. Numeric missing values stay None for the native missing split rule.
No optional dependency is imported. pandas.NA is recognized only when pandas is
already loaded by the caller. Each validation fold needs its own instance.
"""
import copy
import math
import sys
from numbers import Real


def _m26_missing(v):
    if v is None:
        return True
    if isinstance(v, Real):
        return math.isnan(float(v))
    # Never evaluate NA's truth value or import pandas in a standalone runtime.
    pandas = sys.modules.get('pandas')
    return pandas is not None and v is getattr(pandas, 'NA', None)


def _m26_scalar(v):
    if _m26_missing(v):
        return None
    if isinstance(v, str):
        if len(v) > 1024:
            raise ValueError('Category string exceeds 1024 characters')
        return v
    if isinstance(v, Real):
        z = float(v)
        if math.isfinite(z):
            return z
    raise ValueError('Expected finite numeric, string category, None, numeric NaN or pandas.NA; infinity/objects are unsupported')


def _m26_rows(X):
    names = list(X.columns) if hasattr(X, 'columns') else None
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy()
    if hasattr(X, 'tolist'):
        X = X.tolist()
    if not isinstance(X, (list, tuple)) or not 1 <= len(X) <= 100000:
        raise ValueError('X must be a nonempty dense 2D table, at most 100000 rows')
    if not isinstance(X[0], (list, tuple)) or not 1 <= len(X[0]) <= 4096:
        raise ValueError('X must have 1..4096 columns')
    n = len(X[0])
    if len(X)*n > 2000000 or any(not isinstance(r, (list, tuple)) or len(r) != n for r in X):
        raise ValueError('Rectangular X required, at most 2000000 input cells')
    if names is not None and (len(names) != n or any(not isinstance(v, str) or not v or len(v)>1024 for v in names) or len(set(names)) != n):
        raise ValueError('Unique nonempty string column names required')
    return [[_m26_scalar(v) for v in r] for r in X], names


class TrainEncoder:
    """Immutable-by-copy serializable schema; fit accepts TRAIN rows only."""
    def fit(self, X):
        rows, names = _m26_rows(X)
        columns = []
        for col in zip(*rows):
            present = [v for v in col if v is not None]
            if any(isinstance(v, str) for v in present):
                if any(not isinstance(v, str) for v in present):
                    raise ValueError('Mixed numeric/string column; supply a consistent type')
                vocab = sorted(set(present))
                if len(vocab) > 64:
                    raise ValueError('Experimental categorical vocabulary cap is 64 per TRAIN column')
                columns.append({'kind': 'categorical', 'vocabulary': vocab})
            else:
                columns.append({'kind': 'numeric'})
        state = {'schema': 'morph026-train-encoder-v1', 'names': names, 'columns': columns}
        checked = self.from_dict(state)  # Validate before replacing a fitted schema.
        self.state_, self.width_ = checked.state_, checked.width_
        return self

    def to_dict(self):
        if not hasattr(self, 'state_'):
            raise RuntimeError('Encoder is not fitted')
        return copy.deepcopy(self.state_)

    @classmethod
    def from_dict(cls, state):
        if type(state) is not dict or set(state) != {'schema', 'names', 'columns'} or state['schema'] != 'morph026-train-encoder-v1':
            raise ValueError('Exact encoder schema required')
        cols, names = state['columns'], state['names']
        if type(cols) is not list or not 1 <= len(cols) <= 4096:
            raise ValueError('Bounded column schema required')
        if names is not None and (type(names) is not list or len(names)!=len(cols) or any(not isinstance(n,str) or not n or len(n)>1024 for n in names) or len(set(names))!=len(names)):
            raise ValueError('Invalid column names')
        width = 0
        for c in cols:
            if c == {'kind': 'numeric'}:
                width += 1
            elif type(c) is dict and set(c)=={'kind','vocabulary'} and c['kind']=='categorical':
                v = c['vocabulary']
                if type(v) is not list or not 1<=len(v)<=64 or any(not isinstance(s,str) or len(s)>1024 for s in v) or v!=sorted(set(v)):
                    raise ValueError('Bounded canonical TRAIN vocabulary required')
                width += len(v)+2
            else:
                raise ValueError('Unknown encoder column rule')
        if width>4096:
            raise ValueError('Encoded feature cap is 4096')
        obj=cls(); obj.state_=copy.deepcopy(state); obj.width_=width
        return obj

    def transform(self, X):
        s=self.to_dict(); rows,names=_m26_rows(X)
        if len(rows[0])!=len(s['columns']):
            raise ValueError('Missing or extra input columns')
        if s['names'] is not None:
            if names is None or set(names)!=set(s['names']):
                raise ValueError('Named training schema requires the same named columns')
            order=[names.index(n) for n in s['names']]
            rows=[[r[i] for i in order] for r in rows]
        elif names is not None:
            raise ValueError('Training used positional columns; named prediction input is ambiguous')
        result=[]
        for row in rows:
            out=[]
            for value,c in zip(row,s['columns']):
                if c['kind']=='numeric':
                    if isinstance(value,str):
                        raise ValueError('String supplied to numeric TRAIN column')
                    out.append(value)
                else:
                    if value is not None and not isinstance(value,str):
                        raise ValueError('Numeric supplied to string categorical column')
                    vocab=c['vocabulary']; z=[0.0]*(len(vocab)+2)
                    i=len(vocab) if value is None else vocab.index(value) if value in vocab else len(vocab)+1
                    z[i]=1.0;out.extend(z)
            result.append(out)
        return result
