import copy, math, json, tempfile, os
from ..temporary import OwnedTemporaryDirectory
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
from .base import Base as Previous, PARAMS
from .preprocessing import TrainEncoder
from .runtime_io import require, explicit_input, configure

class VariantUnavailable(ValueError):
    pass

class Control(Previous):

    def __init__(self, *, task='auto', families='auto', morphing='auto', tuning='auto', preset='balanced', metric='auto', time_budget='auto', n_trials='auto', n_jobs='auto', random_state=None, verbose=0, capacity=None):
        self.capacity = copy.deepcopy(capacity)
        super().__init__(task=task, families=families, morphing=morphing, tuning=tuning, preset=preset, metric=metric, time_budget=time_budget, n_trials=n_trials, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    def _validate(self):
        super()._validate()
        if self.capacity is not None:
            if type(self.capacity) is not dict or not set(self.capacity) <= {'max_leaves', 'max_depth', 'n_estimators'}:
                raise ValueError('capacity supports finite max_leaves/max_depth/n_estimators; no inactive options')
            for name, value in self.capacity.items():
                if type(value) is not int or not 1 <= value <= {'max_leaves': 16, 'max_depth': 8, 'n_estimators': 8}[name]:
                    raise ValueError('Finite integer structural capacity required')

    def get_params(self, deep=True):
        return copy.deepcopy({k: getattr(self, k) for k in (*PARAMS, 'capacity')})

    def set_params(self, **values):
        if set(values) - set(self.get_params()):
            raise ValueError('Unknown estimator parameter')
        new = type(self)(**self.get_params() | values)
        self.__dict__ = new.__dict__
        return self

    def report(self):
        value = super().report()
        value['variants'] = self.available_variants()
        return value

    def available_variants(self):
        self._fitted()
        return {k: {'origin': v['origin'], 'params': copy.deepcopy(v['recipe']['params']), 'exportable': v['exportable'], 'evidence': v['evidence'], 'SEARCH_primary': v.get('SEARCH_primary')} for k, v in self._variants.items()}

    def _variant(self, name):
        self._fitted()
        name = self._selected_variant if name == 'selected' else name
        if name not in self._variants:
            raise VariantUnavailable('Requested variant has no retained, verified weights; a separately budgeted fit may be needed')
        v = self._variants[name]
        if not v['exportable']:
            raise VariantUnavailable('Software replay cannot be exported as a newly trained pure/tuned variant')
        obj = Previous(**{k: v for k, v in self.get_params().items() if k != 'capacity'})
        for k in ('encoder_', 'task_', 'n_features_in_', 'classes_', 'feature_names_in_'):
            if hasattr(self, k):
                setattr(obj, k, copy.deepcopy(getattr(self, k)))
        obj.model_ = v['model']
        obj.best_recipe_ = copy.deepcopy(v['recipe'])
        obj.best_family_ = v['recipe']['family']
        obj.best_params_ = copy.deepcopy(v['recipe']['params'])
        obj.report_ = self.report()
        obj.report_.update(exported_variant=name, SEARCH_primary=v.get('SEARCH_primary'), best_family=obj.best_family_)
        obj.stop_reason_ = self.stop_reason_
        return obj

    def export_python(self, path, *, variant='selected'):
        self._variant(variant).export_python(path)

    def save_recipe(self, path, *, variant='selected'):
        self._variant(variant).save_recipe(path)

    def save(self, path, *, variant='all'):
        self._fitted()
        configure_temp()
        names = list(self._variants) if variant == 'all' else [self._selected_variant if variant == 'selected' else variant]
        packed = {}
        origins = {}
        with OwnedTemporaryDirectory() as tmp:
            for name in names:
                obj = self._variant(name)
                p = artifact_path(Path(tmp) / (name + '.json'))
                obj.save(p)
                packed[name] = read_json(p, LoadLimits(max_bytes=32 * 1024 ** 2))
                origins[name] = {'origin': self._variants[name]['origin'], 'evidence': self._variants[name]['evidence']}
        body = {'schema': {'comp009-frozen-form-v1': 'comp009-native-control-v1', 'comp003-frozen-form-v1': 'comp003-native-control-v1', 'comp002-frozen-form-v1': 'comp002-native-control-v1', 'product002-frozen-form-v1': 'product002-native-control-v1', 'product003-frozen-form-v1': 'product003-native-control-v1', 'product006-frozen-form-v1': 'product006-native-control-v1'}.get(next(iter(packed.values()))['recipe']['schema'], 'morph027-native-control-v1'), 'constructor': self.get_params(), 'selected_variant': self._selected_variant if self._selected_variant in names else names[0], 'variants': packed, 'origins': origins}
        if len(json.dumps(body, allow_nan=False).encode('utf8')) > 32 * 1024 ** 2:
            raise ValueError('Bounded total variant payload required')
        path = artifact_path(path)
        if path.exists():
            raise FileExistsError('Never overwrite research model evidence')
        write_json(path, body)

    @classmethod
    def load(cls, path):
        data = read_json(artifact_path(path), LoadLimits(max_bytes=32 * 1024 ** 2))
        if set(data) != {'schema', 'constructor', 'selected_variant', 'variants', 'origins'} or data['schema'] not in ('morph027-native-control-v1', 'comp002-native-control-v1', 'comp003-native-control-v1', 'comp009-native-control-v1', 'product002-native-control-v1', 'product003-native-control-v1', 'product006-native-control-v1'):
            raise ValueError('Exact bounded private control schema required')
        vs = data['variants']
        origins = data['origins']
        if type(vs) is not dict or not vs or (not set(vs) <= {'pure', 'tuned'}) or (set(origins) != set(vs)) or (data['selected_variant'] not in vs):
            raise ValueError('Bounded named variants required')
        obj = cls(**data['constructor'])
        configure_temp()
        loaded = {}
        with OwnedTemporaryDirectory() as tmp:
            for name, body in vs.items():
                if type(body) is not dict:
                    raise ValueError('Each model variant must be a JSON object')
                o = origins[name]
                if {'morph027-native-control-v1': 'morph026-native-estimator-v1', 'comp002-native-control-v1': 'comp002-native-estimator-v1', 'comp009-native-control-v1': 'comp009-native-estimator-v1', 'comp003-native-control-v1': 'comp003-native-estimator-v1', 'product002-native-control-v1': 'product002-native-estimator-v1', 'product003-native-control-v1': 'product003-native-estimator-v1', 'product006-native-control-v1': 'product006-native-estimator-v1'}[data['schema']] != body.get('schema'):
                    raise ValueError('Mixed experimental envelope versions')
                if data['schema'] == 'morph027-native-control-v1' and 'COMP002_' in str(o):
                    raise ValueError('Never relabel fixture as historical training')
                if set(o) != {'origin', 'evidence'} or o['origin'] not in ('PRE_FINAL_NUMERIC_SEARCH', 'VERIFIED_B0_UNTUNED', 'POST_FINAL_NUMERIC_SEARCH') or (not isinstance(o['evidence'], str)) or ('MOCK' in o['evidence']):
                    raise ValueError('Explicit genuine variant provenance required')
                if name == 'pure' and o['origin'] == 'POST_FINAL_NUMERIC_SEARCH' or (name == 'tuned' and o['origin'] != 'POST_FINAL_NUMERIC_SEARCH'):
                    raise ValueError('Never relabel tuned weights as pure')
                p = artifact_path(Path(tmp) / (name + '.json'))
                write_json(p, body)
                v = Previous.load(p)
                if v.task_ != body['recipe']['task']:
                    raise ValueError('Changed task')
                expected = HyperParams(**v.best_params_)
                for m in v.model_.components:
                    actual = asdict(m.params_)
                    want = asdict(expected if v.best_family_ == 'boost' else replace(expected, n_estimators=1, learning_rate=1.0, row_fraction=1.0))
                    if actual != want:
                        raise ValueError('Recipe parameters do not describe the actual weights')
                if name == 'tuned' and (not v.report_.get('trials') or not _numeric_evidence(data['schema'], o, v.report_)):
                    raise ValueError('No executed numeric trajectory for tuned variant')
                loaded[name] = v
        v = loaded[data['selected_variant']]
        for other in loaded.values():
            if (other.task_, other.classes_ if hasattr(other, 'classes_') else (), other.best_recipe_['structure'], other.best_recipe_['training_identity']) != (v.task_, v.classes_ if hasattr(v, 'classes_') else (), v.best_recipe_['structure'], v.best_recipe_['training_identity']):
                raise ValueError('Variants must share the same frozen structure/data/map')
        for k in ('model_', 'encoder_', 'task_', 'n_features_in_', 'classes_', 'feature_names_in_', 'best_family_', 'best_recipe_', 'best_params_', 'stop_reason_', 'report_'):
            if hasattr(v, k):
                setattr(obj, k, getattr(v, k))
        obj.structure_ = copy.deepcopy(v.best_recipe_['structure'])
        obj._selected_variant = data['selected_variant']
        obj._variants = {k: {'model': v.model_, 'recipe': copy.deepcopy(v.best_recipe_), **origins[k], 'exportable': True, 'SEARCH_primary': v.report_.get('SEARCH_primary')} for k, v in loaded.items()}
        return obj

    def export_bundle(self, directory, *, variants=('pure',)):
        if type(variants) is not tuple or not variants or len(set(variants)) != len(variants):
            raise ValueError('Distinct variant tuple required')
        for name in variants:
            self._variant(name)
        directory = artifact_path(directory)
        if directory.exists():
            raise FileExistsError('New export directory required')
        directory.mkdir(parents=True)
        out = []
        for name in variants:
            p = directory / (name + '.json')
            self.save(p, variant=name)
            out.append(reference(p))
            p = directory / (name + '-recipe.json')
            self.save_recipe(p, variant=name)
            out.append(reference(p))
            p = directory / (name + '-predictor.py')
            self.export_python(p, variant=name)
            out.append(reference(p))
        p = directory / 'EXPORT.json'
        write_json(p, {'schema': 'morph027-inference-export-v1', 'status': 'COMPLETE', 'artifacts': out, 'retraining_procedure': 'NOT_SUPPORTED', 'new_fits': 0, 'report': self.report()})
        return reference(p)

    def export_retrainable(self, path):
        raise NotImplementedError('Only inference standalone and declarative recipe are qualified; generic retraining admission is missing')

class MorphTree(Control):
    """Installed retained-model facade. No fresh training implementation."""

    @classmethod
    def load(cls, path):
        source = explicit_input(path)
        data = read_json(source, LoadLimits(max_bytes=32 * 1024 ** 2))
        if data.get('schema') == 'morph027-native-control-v1':
            obj = super().load(source)
        else:
            v = Previous.load(source)
            expected = HyperParams(**v.best_params_)
            for member in v.model_.components:
                want = expected if v.best_family_ == 'boost' else replace(expected, n_estimators=1, learning_rate=1.0, row_fraction=1.0)
                if asdict(member.params_) != asdict(want):
                    raise ValueError('Recipe versus actual retained native parameters')
            obj = cls(**v.get_params())
            obj.__dict__.update(v.__dict__)
            obj.capacity = None
            obj._selected_variant = 'pure'
            obj._variants = {'pure': {'model': v.model_, 'recipe': copy.deepcopy(v.best_recipe_), 'origin': 'VERIFIED_B0_UNTUNED', 'evidence': 'EXPLICIT_SUPPLIED_RETAINED_INPUT_NOT_NEW_FIT', 'exportable': True, 'SEARCH_primary': v.report_.get('SEARCH_primary')}}
        engine, r = require(('load', 'replay'))
        obj._admitted_component = next((v for v in r['components'] if artifact_path(engine.q['input_root']) / v['path'] == artifact_path(path)))
        obj._check_admission(('load', 'replay'))
        return obj

    def _check_admission(self, operations):
        e, r = require(operations)
        actual = {'family': self.best_family_, 'members': len(self.model_.components), 'stages': self.best_params_['n_estimators'] if self.best_family_ == 'boost' else 1, 'classes': len(getattr(self, 'classes_', ())) or 1, 'genome_sha256': digest(self.best_recipe_['structure']['genome'])}
        if r['task'] != self.task_ or r['recipe'] != actual or self._admitted_component not in r['components']:
            raise ValueError('Model/request actual task, form or component mismatch')

    def fit(self, X, y):
        self.last_attempt_ = {'status': 'BLOCKED', 'new_fits': 0, 'new_trials': 0, 'reason': 'NEW_INSTALLED_TRAINING_REQUIRES_QUALIFIED_BOUNDARY_AND_NEW_OWNER_CONSENT'}
        raise AdmissionError(self.last_attempt_)

    def predict(self, X):
        self._check_admission(('infer', 'replay'))
        return super().predict(X)

    def predict_proba(self, X):
        self._check_admission(('infer', 'replay'))
        return super().predict_proba(X)

    def save(self, path, *, variant='all'):
        self._check_admission(('save',))
        e, r = require(('save',))
        super().save(path, variant=variant)
        return e.record_output(path)

    def save_recipe(self, path, *, variant='selected'):
        self._check_admission(('export',))
        e, r = require(('export',))
        super().save_recipe(path, variant=variant)
        return e.record_output(path)

    def export_python(self, path, *, variant='selected'):
        self._check_admission(('export',))
        e, r = require(('export',))
        super().export_python(path, variant=variant)
        return e.record_output(path)

    def export_bundle(self, *a, **k):
        raise NotImplementedError('Use separately admitted save/export operations; no broad bundle capability')

    def recipe(self):
        self._fitted()
        return copy.deepcopy(self.best_recipe_)

def _numeric_evidence(schema, origin, report):
    if schema=='product006-native-control-v1':
        rows=[e for e in report.get('events',[]) if e.get('operation')=='COMP003_ACCOUNTING_AND_SCOPE']
        if len(rows)!=1:return False
        row=rows[0];counts=row['counts'];n=len(report.get('trials',[]))
        if not (origin['evidence']=='USER_API_NATIVE_EXECUTION_NOT_SCIENTIFIC_CONFIRMATION'
                and row['details']['trial_evidence']=='USER_DATA_REAL_OPTUNA_NOT_SCIENTIFIC_CONFIRMATION'
                and 1<=n<=4 and n<=counts['Optuna_trials']<=4):return False
        if counts['Optuna_trials']==n and counts['physical_fits_started']==counts['physical_fits_completed']:return True
        p=report.get('morphing',{}).get('verified_report',{})
        recovery=p.get('budget_recovery',{})
        if (recovery.get('schema')!='product006-verified-budget-recovery-v1'
                or recovery.get('phase_completion')!='PARTIAL_BUDGET_VALID_ANCHOR_RETURNED'
                or p.get('counts',{}).get('real_trials_completed')!=n
                or p['counts'].get('real_trials_created')!=counts['Optuna_trials']):return False
        abandoned=recovery.get('unadopted_components',{})
        incomplete=counts['physical_fits_started']-counts['physical_fits_completed']
        return (incomplete>=0 and incomplete==sum(not x['native_completed'] for x in abandoned.values())
                and all(x.get('used_in_final_model') is False for x in abandoned.values())
                and any(e.get('kind')=='SEARCH_STOPPED_BUDGET' for e in p.get('events',[])))

    if schema == 'product003-native-control-v1':
        rows = [e for e in report.get('events', []) if e.get('operation') == 'COMP003_ACCOUNTING_AND_SCOPE']
        if len(rows) != 1:
            return False
        row = rows[0]
        counts = row['counts']
        n = len(report.get('trials', []))
        return origin['evidence'] == 'USER_API_NATIVE_EXECUTION_NOT_SCIENTIFIC_CONFIRMATION' and row['details']['trial_evidence'] == 'USER_DATA_REAL_OPTUNA_NOT_SCIENTIFIC_CONFIRMATION' and (1 <= n <= 4) and (counts['Optuna_trials'] == n) and (counts['physical_fits_started'] == counts['physical_fits_completed'])
    if schema != 'comp003-native-control-v1':
        return report.get('trial_evidence') in ('EXECUTOR_BOUND_NUMERIC_CALLBACKS', 'AUTHENTICATED_HISTORICAL_REAL_TPE4')
    rows = [e for e in report.get('events', []) if e.get('operation') == 'COMP003_ACCOUNTING_AND_SCOPE']
    if len(rows) != 1:
        return False
    e = rows[0]
    kind = e['details']['trial_evidence']
    counts = e['counts']
    n = len(report['trials'])
    if 'SOFTWARE_FIXTURE' in origin['evidence']:
        return kind == 'SIMULATED_OPTUNA_FIXTURE_NOT_REAL_TRIALS' and counts['trials'] == 0 and (counts['simulated_trials'] == n) and (n == 2)
    return origin['evidence'] == 'COMP003_EXACT_OWNER_ADMITTED_NATIVE_TRAINING' and kind == 'EXACT_OWNER_BOUND_REAL_OPTUNA_TRIALS' and (counts['trials'] == n) and (n == 2) and (counts['simulated_trials'] == 0)
