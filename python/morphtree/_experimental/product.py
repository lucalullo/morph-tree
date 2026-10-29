"""Installed integrated consumer of explicitly supplied, authenticated models.

Packaged with relative imports; not imported as a workspace learner. SHA is a
caller-supplied custody identity, not a claim that a model is scientifically good.
"""
import copy
import hashlib
import os
from pathlib import Path
from dataclasses import asdict, replace
from morphtree.config import HyperParams
from morphtree.persistence import read_json, LoadLimits
from .admission import AdmissionError, digest
from .runtime_io import configure, artifact_path, publish
from .base import Base, _payloads
from .facade import Control
from .roundtrip_v3 import decode
from .support import FamilyChoice
from .learning import mutation_admission

class MorphTree(Control):
    """Unified experimental API. Fresh fit needs a new qualified v2 admission.

    Use load(model, sha256=..., workdir=...) for a supplied retained model.
    Files remain declarative and bounded; no code or data are loaded implicitly.
    """

    def fit(self, X, y):
        self.last_attempt_ = {'status': 'BLOCKED', 'physical_fits': 0, 'trials': 0, 'reason': 'COMPLETION_TRAINING_BOUNDARY_REQUIRES_QUALIFICATION_AND_NEW_OWNER_CONSENT'}
        raise AdmissionError(self.last_attempt_)

    @classmethod
    def load(cls, path, *, sha256, workdir):
        configure(workdir)
        source = Path(path).resolve()
        if not source.is_file() or not 0 < source.stat().st_size <= 32 * 1024 ** 2:
            raise ValueError('Explicit bounded model input required')
        with source.open('rb') as stream:raw=stream.read(32*1024**2+1)
        if not 0<len(raw)<=32*1024**2:raise ValueError('Explicit bounded model input required')
        data = decode(raw, sha256)
        if type(data) is not dict:raise ValueError('Model input must be a JSON object')
        folder = artifact_path('.inputs')
        folder.mkdir(exist_ok=True)
        snapshot = artifact_path(folder / (sha256 + '.json'))
        if snapshot.exists():
            if snapshot.read_bytes() != raw:
                raise ValueError('Corrupt retained input snapshot')
        else:
            pending = artifact_path(str(snapshot) + '.pending')
            with pending.open('xb') as out:
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            if snapshot.exists():
                raise FileExistsError('Concurrent input snapshot')
            os.rename(pending, snapshot)
        if data.get('schema') in ('morph027-native-control-v1', 'comp002-native-control-v1', 'comp003-native-control-v1', 'comp009-native-control-v1', 'product002-native-control-v1', 'product003-native-control-v1', 'product006-native-control-v1'):
            old = Control.load(snapshot)
        else:
            old = Base.load(snapshot)
            if old.report_.get('tuning') != 'DISABLED' or old.report_.get('trials'):
                raise ValueError('Legacy input must have an explicitly untuned report; use verified variants otherwise')
            old._selected_variant = 'pure'
            old._variants = {'pure': {'model': old.model_, 'recipe': copy.deepcopy(old.best_recipe_), 'origin': 'PRE_FINAL_NUMERIC_SEARCH', 'evidence': 'EXPLICIT_RETAINED_WEIGHTS_NOT_NEW_FIT', 'exportable': True, 'SEARCH_primary': old.report_.get('SEARCH_primary')}}
        for variant in old._variants.values():
            recipe = variant['recipe']
            expected = HyperParams(**recipe['params'])
            for member in variant['model'].components:
                want = expected if recipe['family'] == 'boost' else replace(expected, n_estimators=1, learning_rate=1.0, row_fraction=1.0)
                if asdict(member.params_) != asdict(want):
                    raise ValueError('Recipe does not describe actual native parameters')
        obj = cls(**old.get_params())
        obj.__dict__.update(old.__dict__)
        obj.capacity = getattr(old, 'capacity', None)
        obj._workdir = str(Path(workdir).resolve())
        obj._input_sha256 = sha256
        return obj

    def _io(self):
        self._fitted()
        configure(self._workdir)

    def predict(self, X):
        self._io()
        return super().predict(X)

    def predict_proba(self, X):
        self._io()
        return super().predict_proba(X)

    def recipe(self):
        self._fitted()
        return copy.deepcopy(self.best_recipe_)

    def cost_report(self):
        self._fitted()
        return {'current_physical_fits': 0, 'current_trials': 0, 'current_OOF': 0, 'scope': 'RETAINED_MODEL_CONSUMPTION_ONLY', 'historical_learning_cost': 'SEE_ORIGINAL_REPORT_NOT_REMEASURED', 'source_artifact_sha256': self._input_sha256, 'peak_RSS': 'UNKNOWN', 'performance_qualified': False}

    def save(self, path, *, variant='all'):
        self._io()
        return super().save(path, variant=variant)

    def save_recipe(self, path, *, variant='selected'):
        self._io()
        if artifact_path(path).exists():
            raise FileExistsError('Never replace a completed recipe')
        return super().save_recipe(path, variant=variant)

    def export_python(self, path, *, variant='selected'):
        self._io()
        return super().export_python(path, variant=variant)

    def export_bundle(self, directory, *, variants=('pure',)):
        self._io()
        return super().export_bundle(directory, variants=variants)

    @classmethod
    def select_retained(cls, anchor, candidate, *, X, y, ids, role='SEARCH'):
        """Compare two existing families on their exact historical SEARCH.

        No new labels, new split, fabricated provenance, fit, OOF or tuning.
        This replay measures selection correctness, never policy utility.
        """
        if not isinstance(anchor, cls) or not isinstance(candidate, cls):
            raise ValueError('Explicit authenticated model objects required')
        anchor._io()
        candidate._fitted()
        a, b = (anchor.recipe(), candidate.recipe())
        if role != 'SEARCH' or anchor.task_ != candidate.task_ or a['training_identity'] != b['training_identity']:
            raise ValueError('Same actual TRAIN/SEARCH task identity required')
        if a['preprocessing'] != b['preprocessing'] or getattr(anchor, 'classes_', ()) != getattr(candidate, 'classes_', ()):
            raise ValueError('Same encoding and class vocabulary required')
        if type(ids) is not list or not ids or len(set(ids)) != len(ids) or (len(ids) != len(y)):
            raise ValueError('Unique aligned SEARCH IDs required')
        targets = list(y) if anchor.task_ == 'regression' else [list(anchor.classes_).index(v) for v in y]
        search = {'role': role, 'ids': list(ids), 'X': anchor.encoder_.transform(X), 'y': targets}
        if digest(search) != a['training_identity'].get('SEARCH'):
            raise ValueError('Supplied rows/labels/IDs differ from frozen SEARCH')
        models = {'anchor': anchor, 'candidate': candidate}

        def packet(name, model):
            r = model.recipe()
            return {'id': name, 'task': model.task_, 'family': r['family'], 'recipe': r['structure'], 'learning_identity': r['training_identity'], 'params': r['params'], 'artifact_sha256': digest(_payloads(model.model_))}
        packets = {n: packet(n, m) for n, m in models.items()}
        events = []

        def verify(p, selection):
            model = models[p['id']]
            if p != packets[p['id']] or selection != search or digest(_payloads(model.model_)) != p['artifact_sha256']:
                raise ValueError('Changed verified candidate/SEARCH')
            return model.model_.probabilities_or_values(selection['X'])
        chooser = FamilyChoice(packets['anchor'], search, checkpoint=events.append, verify=verify)
        result = chooser.run(packets['candidate'])
        chosen = models[result['packet']['id']]
        decision = {'schema': 'morphtree-completion-retained-selection-v1', 'selected': result['packet']['id'], 'anchor_loss': result['anchor_loss'], 'selected_loss': result['loss'], 'events': events, 'decision': 'ACCEPT_OTHER_FAMILY' if chosen is candidate else 'RETAIN_ANCHOR', 'morphing': mutation_admission('auto'), 'structure_frozen': chosen.recipe(), 'physical_fits': 0, 'trials': 0, 'GUARD': 0, 'evidence': 'RETAINED_REPLAY_NOT_TRAINED_ADAPTIVE_UTILITY'}
        return (chosen, decision)
