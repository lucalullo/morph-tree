import copy, math, json, tempfile, os
from pathlib import Path
from dataclasses import asdict, replace
from morphtree.config import AlgorithmGenome, HyperParams
from morphtree.persistence import read_json, write_json, LoadLimits
from morphtree.model import MorphTreeClassifier, MorphTreeRegressor, MorphTreeMulticlassClassifier
from morphtree.metrics import point_losses
from .admission import digest, sha, AdmissionError
from .runtime_io import artifact_path, configure_temp, reference, publish
NATIVE = {'binary': MorphTreeClassifier, 'multiclass': MorphTreeMulticlassClassifier, 'regression': MorphTreeRegressor}
TASKS = ('binary', 'multiclass', 'regression')
from .support import FamilyModel
from .preprocessing import TrainEncoder, _m26_rows, _m26_scalar
PARAMS = ('task', 'families', 'morphing', 'tuning', 'preset', 'metric', 'time_budget', 'n_trials', 'n_jobs', 'random_state', 'verbose')

def _payloads(model):
    configure_temp()
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(model.components):
            p = artifact_path(Path(tmp) / ('member-' + str(i) + '.json'))
            m.save_model(p)
            out.append(read_json(p))
    return out

def _load_members(task, payloads):
    if task not in NATIVE or type(payloads) is not list or (not 1 <= len(payloads) <= 32):
        raise ValueError('Bounded known native family required')
    if any(type(payload) is not dict for payload in payloads):
        raise ValueError('Each native model payload must be a JSON object')
    models = [NATIVE[task]._from_payload(p, limits=LoadLimits(max_bytes=16 * 1024 ** 2), backend='pure') for p in payloads]
    return FamilyModel(task, models, [1 / len(models)] * len(models))

class Base:

    def __init__(self, *, task='auto', families='auto', morphing='auto', tuning='auto', preset='balanced', metric='auto', time_budget='auto', n_trials='auto', n_jobs='auto', random_state=None, verbose=0):
        for k, v in locals().copy().items():
            if k != 'self':
                setattr(self, k, v)
        self._validate()

    def _validate(self):
        if self.task not in ('auto', 'binary', 'multiclass', 'regression') or self.morphing not in ('auto', 'none') or self.tuning not in ('auto', 'none', 'optuna') or (self.preset != 'balanced'):
            raise ValueError('Unsupported task/morphing/tuning/preset')
        if self.families not in ('auto', 'boost', 'bag'):
            raise ValueError('families must be auto, boost or bag')
        if self.metric not in ('auto', 'logloss', 'mse'):
            raise ValueError('Only task-consistent logloss/mse primary metrics are supported')
        if self.n_jobs != 'auto' and (type(self.n_jobs) is not int or self.n_jobs != 1):
            raise ValueError('Experimental serial scheduler supports n_jobs=1 or auto only')
        if self.n_trials != 'auto' and (type(self.n_trials) is not int or not 1 <= self.n_trials <= 4):
            raise ValueError('n_trials must be auto or 1..4')
        if self.time_budget != 'auto' and (type(self.time_budget) not in (int, float) or not math.isfinite(self.time_budget) or (not 0 < self.time_budget <= 300)):
            raise ValueError('Finite 0<time_budget<=300 required')
        if self.random_state is not None and (type(self.random_state) is not int or not 0 <= self.random_state < 2 ** 31):
            raise ValueError('random_state must be None or an integer in [0,2**31)')
        if type(self.verbose) is not int or self.verbose not in (0, 1):
            raise ValueError('verbose must be 0 or 1')

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in PARAMS}

    def set_params(self, **values):
        if set(values) - set(PARAMS):
            raise ValueError('Unknown estimator parameter')
        new = type(self)(**self.get_params() | values)
        self.__dict__ = new.__dict__
        return self

    def _fitted(self):
        if not hasattr(self, 'model_'):
            raise RuntimeError('MorphTree experimental estimator is not fitted')

    def predict_proba(self, X):
        self._fitted()
        if self.task_ == 'regression':
            raise ValueError('predict_proba is unavailable for regression')
        p = self.model_.probabilities_or_values(self.encoder_.transform(X))
        return [[1 - v, v] for v in p] if self.task_ == 'binary' else p

    def predict(self, X):
        self._fitted()
        if self.task_ == 'regression':
            return self.model_.probabilities_or_values(self.encoder_.transform(X))
        return [self.classes_[max(range(len(p)), key=p.__getitem__)] for p in self.predict_proba(X)]

    def report(self):
        self._fitted()
        return copy.deepcopy(self.report_)

    def save(self, path):
        self._fitted()
        path = artifact_path(path)
        data = {'schema': {'comp009-frozen-form-v1': 'comp009-native-estimator-v1', 'comp003-frozen-form-v1': 'comp003-native-estimator-v1', 'comp002-frozen-form-v1': 'comp002-native-estimator-v1', 'product002-frozen-form-v1': 'product002-native-estimator-v1', 'product003-frozen-form-v1': 'product003-native-estimator-v1', 'product006-frozen-form-v1': 'product006-native-estimator-v1'}.get(self.best_recipe_['schema'], 'morph026-native-estimator-v1'), 'constructor': self.get_params(), 'recipe': copy.deepcopy(self.best_recipe_), 'classes': list(getattr(self, 'classes_', ())), 'report': self.report(), 'native_payloads': _payloads(self.model_)}
        if len(json.dumps(data, allow_nan=False).encode('utf8')) > 32 * 1024 ** 2:
            raise ValueError('Aggregate private model cap is 32MiB')
        write_json(path, data)

    def save_recipe(self, path):
        """Declarative frozen form/params, no weights or self-executing training."""
        self._fitted()
        write_json(artifact_path(path), self.best_recipe_)

    def export_python(self, path):
        """Inference only; reuse native standalone runtime and exact preprocessing.

        No fitter, search, Optuna, private TRAIN rows or acquisition credentials
        are embedded. This is not a retrainable procedure or a pure/tuned pair.
        """
        import ast
        import morphtree.exporting as exporting
        from . import preprocessing
        self._fitted()
        payloads = _payloads(self.model_)
        class_name = NATIVE[self.task_].__name__
        text = exporting._bundle_runtime(inference_only=True, task=self.task_) + '\n'
        text += Path(preprocessing.__file__).read_text('utf8') + '\n'
        text += '_M26_ENCODER = TrainEncoder.from_dict(' + repr(self.encoder_.to_dict()) + ')\n'
        text += '_M26_TASK = ' + repr(self.task_) + '\n_M26_CLASSES = ' + repr(tuple(getattr(self, 'classes_', ()))) + '\n'
        text += '_M26_MEMBERS = [' + ','.join((class_name + '._from_payload(json.loads(' + repr(json.dumps(p, allow_nan=False)) + '))' for p in payloads)) + ']\n'
        text += "\ndef _m26_values(X):\n    rows=_M26_ENCODER.transform(X)\n    cols=[m.predict(rows) if _M26_TASK=='regression' else m.predict_proba(rows) for m in _M26_MEMBERS]\n    if _M26_TASK=='regression':return [math.fsum((1/len(cols))*c[i] for c in cols) for i in range(len(rows))]\n    if _M26_TASK=='binary':\n        p=[math.fsum((1/len(cols))*c[i][1] for c in cols) for i in range(len(rows))]\n        return [[1-v,v] for v in p]\n    return [[math.fsum((1/len(cols))*c[i][j] for c in cols) for j in range(len(_M26_CLASSES))] for i in range(len(rows))]\n\ndef predict_proba(X):\n    if _M26_TASK=='regression':raise ValueError('Regression has no probabilities')\n    return _m26_values(X)\n\ndef predict(X):\n    p=_m26_values(X)\n    return p if _M26_TASK=='regression' else [_M26_CLASSES[max(range(len(row)),key=row.__getitem__)] for row in p]\n"
        ast.parse(text)
        if len(text.encode('utf8')) > 32 * 1024 ** 2:
            raise ValueError('Standalone cap is 32MiB')
        from .runtime_io import publish
        publish(path, text, lambda: None)

    @classmethod
    def load(cls, path):
        d = read_json(artifact_path(path), LoadLimits(max_bytes=32 * 1024 ** 2))
        if set(d) != {'schema', 'constructor', 'recipe', 'classes', 'report', 'native_payloads'} or d['schema'] not in ('morph026-native-estimator-v1', 'comp002-native-estimator-v1', 'comp003-native-estimator-v1', 'comp009-native-estimator-v1', 'product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1'):
            raise ValueError('Exact research estimator schema required')
        r = d['recipe']
        expected = {'schema', 'task', 'family', 'structure', 'params', 'seed', 'training_identity', 'preprocessing', 'mutations'}
        if type(r) is not dict or set(r) != expected or r['schema'] != {'morph026-native-estimator-v1': 'morph026-frozen-form-v1', 'comp002-native-estimator-v1': 'comp002-frozen-form-v1', 'comp009-native-estimator-v1': 'comp009-frozen-form-v1', 'comp003-native-estimator-v1': 'comp003-frozen-form-v1', 'product002-native-estimator-v1': 'product002-frozen-form-v1', 'product003-native-estimator-v1': 'product003-frozen-form-v1', 'product006-native-estimator-v1': 'product006-frozen-form-v1'}[d['schema']] or (r['task'] not in NATIVE) or (r['family'] not in ('boost', 'bag')) or (r['mutations'] != []):
            raise ValueError('Unsupported private frozen form')
        if type(r['structure']) is not dict or set(r['structure']) != ({'genome', 'classes', 'members', 'encoder', 'lineage'} if d['schema'] in ('comp003-native-estimator-v1', 'comp009-native-estimator-v1', 'product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1') else {'genome', 'classes', 'members', 'encoder'}) or type(d['report']) is not dict or (not isinstance(d['report'].get('stop_reason'), str)):
            raise ValueError('Exact private structure/report required')
        obj = cls(**d['constructor'])
        encoder = TrainEncoder.from_dict(r['preprocessing'])
        model = _load_members(r['task'], d['native_payloads'])
        if any((m.n_features_in_ != encoder.width_ for m in model.components)) or (r['family'] == 'boost' and len(model.components) != 1) or r['structure']['members'] != len(model.components):
            raise ValueError('Model/encoder/member shape mismatch')
        HyperParams(**r['params'])
        classes = d['classes']
        if type(classes) is not list or any((type(c) not in (str, int, float) or (isinstance(c, str) and len(c) > 1024) for c in classes)) or (r['task'] == 'regression' and classes) or (r['task'] != 'regression' and (not 2 <= len(classes) <= 64 or not (all((isinstance(c, str) for c in classes)) or all((type(c) in (int, float) for c in classes))) or classes != sorted(set(classes)) or (r['task'] == 'binary' and len(classes) != 2))):
            raise ValueError('Bounded target vocabulary required')
        if r['task'] == 'multiclass' and any((list(m.classes_) != list(range(len(classes))) for m in model.components)):
            raise ValueError('Native multiclass column alignment differs')
        if r['structure'].get('encoder') != r['preprocessing'] or r['structure'].get('classes') != list(range(len(classes))) or (not _qualified_genome(d, model)):
            raise ValueError('Changed frozen Base form/encoder/columns')
        obj.model_ = model
        obj.encoder_ = encoder
        obj.task_ = r['task']
        obj.n_features_in_ = len(encoder.state_['columns'])
        obj.best_recipe_ = copy.deepcopy(r)
        obj.best_family_ = r['family']
        obj.best_params_ = copy.deepcopy(r['params'])
        obj.report_ = copy.deepcopy(d['report'])
        obj.stop_reason_ = d['report']['stop_reason']
        if classes:
            obj.classes_ = tuple(classes)
        if encoder.state_['names'] is not None:
            obj.feature_names_in_ = tuple(encoder.state_['names'])
        return obj
Base._execution_check = lambda self: None
Base._record_event = lambda self, event: None

def _qualified_genome(data, model):
    if data['schema'] in ('product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1'):
        from morphtree.config import validate_task_genome
        r = data['recipe']
        g = validate_task_genome(r['task'], AlgorithmGenome(**r['structure']['genome']))
        base = AlgorithmGenome()
        depth = AlgorithmGenome(growth_policy='depthwise')
        if g not in (base, depth):
            return False
        lineage = r['structure']['lineage']
        if g == base:
            if lineage != {}:
                return False
        else:
            keys = {'gene', 'parent_packet_sha256', 'from', 'to', 'reason'}
            if type(lineage) is not dict or set(lineage) != keys or lineage['gene'] != 'growth_policy' or (lineage['from'] != base.to_dict()) or (lineage['to'] != g.to_dict()) or (lineage['reason'] != 'OOF_HEURISTIC_HYPOTHESIS_NOT_CAUSAL_UTILITY') or (type(lineage['parent_packet_sha256']) is not str) or (len(lineage['parent_packet_sha256']) != 64) or any((c not in '0123456789abcdef' for c in lineage['parent_packet_sha256'])):
                return False
        return all((m.genome_.to_dict() == g.to_dict() for m in model.components))
    recipe = data['recipe']
    g = recipe['structure'].get('genome')
    allowed = [AlgorithmGenome().to_dict()]
    if data['schema'] in ('comp002-native-estimator-v1', 'comp003-native-estimator-v1', 'comp009-native-estimator-v1', 'product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1') and recipe['task'] == 'binary':
        allowed.append(AlgorithmGenome(growth_policy='depthwise').to_dict())
    if g not in allowed:
        return False
    if data['schema'] == 'comp009-native-estimator-v1':
        lineage = recipe['structure']['lineage']
        if g == allowed[0]:
            if lineage != {}:
                return False
        elif type(lineage) is not dict or set(lineage) != {'gene', 'parent_packet_sha256', 'from', 'to', 'reason'} or lineage['gene'] != 'growth_policy' or (lineage['from'] != allowed[0]) or (lineage['to'] != g) or (lineage['reason'] != 'OOF_HEURISTIC_HYPOTHESIS_NOT_CAUSAL_UTILITY') or (type(lineage['parent_packet_sha256']) is not str) or (len(lineage['parent_packet_sha256']) != 64) or any((c not in '0123456789abcdef' for c in lineage['parent_packet_sha256'])):
            return False
    if data['schema'] == 'comp003-native-estimator-v1':
        lineage = recipe['structure']['lineage']
        if g == allowed[0]:
            if lineage != {}:
                return False
        elif type(lineage) is not dict or set(lineage) != {'gene', 'parent_packet_sha256', 'from', 'to', 'reason'} or lineage['gene'] != 'growth_policy' or (lineage['from'] != allowed[0]) or (lineage['to'] != g) or (lineage['reason'] != 'STATIC_TECHNICAL_NOMINATION_NOT_DIAGNOSTIC_UTILITY') or (type(lineage['parent_packet_sha256']) is not str) or (len(lineage['parent_packet_sha256']) != 64) or any((c not in '0123456789abcdef' for c in lineage['parent_packet_sha256'])):
            return False
    return all((m.genome_.to_dict() == g for m in model.components)) if recipe['task'] == 'binary' else g == allowed[0]
