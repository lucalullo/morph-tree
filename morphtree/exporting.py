"""Readable standalone Python export with a standard-library runtime."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from .config import SPEC_VERSION, VERSION
from .persistence import atomic_text

_RUNTIME_MODULES = ("config", "public", "data", "metrics", "backend", "tree", "diagnostics", "control", "booster", "mutations", "tuning",
                    "persistence", "species", "model", "exporting")


_STANDALONE_HEADER = '"""MorphTree standalone standard-library runtime.\n\nMIT License\n\nCopyright (c) 2026 Luca Lullo\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n"""'


class _RemoveLocalImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.level or node.module == "__future__":
            return None
        return node


def _bundle_runtime(inference_only=False, task='binary'):
    source_file = Path(__file__).resolve()
    current = source_file.read_text(encoding="utf-8")
    marker = "\n# MORPHTREE_" + "BUNDLE_END\n"
    if marker in current:
        return current.split(marker, 1)[0]
    directory = source_file.parent
    if not all((directory / f"{name}.py").is_file() for name in _RUNTIME_MODULES):
        raise RuntimeError("Runtime source files are not available for export")
    blocks = [_STANDALONE_HEADER, "from __future__ import annotations\n"]
    for name in _RUNTIME_MODULES:
        if inference_only and name in ('diagnostics', 'control', 'mutations', 'tuning', 'exporting'):
            continue
        source = (directory / f"{name}.py").read_text(encoding="utf-8")
        tree = _RemoveLocalImports().visit(ast.parse(source))
        if inference_only:
            retained = {
                'MorphTreeClassifier': {'__init__','_validate_backend_config','_make_backend','_require_fitted',
                                        'predict_proba','predict','score','load_model','_from_payload'},
                'MorphTreeRegressor': {'__init__','_make_backend','_require_fitted','predict','score','load_model','_from_payload'},
                'MorphTreeMulticlassClassifier': {'__init__','_make_backend','_require_fitted','predict_proba','predict','score','load_model','_from_payload'},
                'BinaryHistogramBooster': {'__init__','predict_raw','predict_proba'},
                'HistogramTree': {'__init__','predict_binned'},
                'PythonBackend': {'bin','probabilities','predict_tree','add_update','multiclass_raw','multiclass_probabilities',
                                  'multiclass_update','center_multiclass'},
                'Binner': {'__init__','transform'},
                'AlgorithmSpecies': {'__post_init__','structural_fingerprint'},
            }
            excluded = {'TrainingCache', 'AutoMorphTree'} | ({'MorphTreeRegressor'} if task == 'binary' else set())
            tree.body = [node for node in tree.body if not isinstance(node, ast.ClassDef) or node.name not in excluded]
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in retained:
                    node.body = [part for part in node.body if not isinstance(part, (ast.FunctionDef, ast.Assign))
                                 or isinstance(part, ast.FunctionDef) and part.name in retained[node.name]]
                    # AlgorithmSpecies dataclass annotations are retained by the above filter.
        ast.fix_missing_locations(tree)
        blocks.append(f"\n# Module: {name}\n" + ast.unparse(tree))
    return "\n\n".join(blocks)


def export_algorithm(species, path):
    payload = json.dumps(species.to_dict(), sort_keys=True, allow_nan=False)
    suffix = '''
_SPECIES_DATA = json.loads(PAYLOAD_LITERAL)
_EXPORTED_GENOME, _EXPORTED_PARAMS, _EXPORTED_METADATA = decode_species(_SPECIES_DATA)

class DiscoveredAlgorithm(MorphTreeClassifier):
    """Train this frozen algorithm on new data. Optional native numeric retuning."""
    def __init__(self, *, params=None, config=None, retune=False, genome=None, backend='pure'):
        if backend != 'pure':
            raise ValueError('Standalone algorithm supports only pure backend')
        if type(retune) is not bool:
            raise ValueError("retune must be a bool")
        if config is not None and not isinstance(config, MorphTreeConfig):
            raise ValueError("config must be a MorphTreeConfig")
        if genome is not None:
            if not isinstance(genome, AlgorithmGenome) or genome.fingerprint() != _EXPORTED_GENOME.fingerprint():
                raise ValueError("This exported algorithm has a frozen structural fingerprint; genome mismatch")
        recipe = _EXPORTED_METADATA.get('learning_recipe')
        settings = replace(recipe_settings(recipe, config),
                           enable_morphing=False, enable_tuning=retune, tuner="native")
        if recipe is None and config is None:
            settings = replace(settings, memory_enabled=True)
        super().__init__(config=settings, genome=_EXPORTED_GENOME, params=_EXPORTED_PARAMS if params is None else params)
        self._algorithm_locked = True
        self._training_partition = recipe['partition'] if recipe else None

    @classmethod
    def from_algorithm(cls, algorithm, *, retune=False, config=None):
        species = algorithm if isinstance(algorithm, AlgorithmSpecies) else AlgorithmSpecies.load(algorithm)
        return cls(params=species.parameters, genome=species.genome, config=config, retune=retune)

    @classmethod
    def discover(cls, *args, **kwargs):
        raise ValueError("DiscoveredAlgorithm is frozen; use MorphTreeClassifier.discover() for a new structural search")

SPECIES = AlgorithmSpecies(_EXPORTED_GENOME, _EXPORTED_PARAMS, _EXPORTED_METADATA)
'''.replace("PAYLOAD_LITERAL", repr(payload))
    atomic_text(path, _bundle_runtime() + "\n# MORPHTREE_BUNDLE_END\n" + suffix)


def export_predictor(model, path):
    if getattr(model, 'task', None) == 'multiclass':
        text = json.dumps(model._model_payload(include_report=False), sort_keys=True, allow_nan=False)
        suffix = '''
_MODEL = MorphTreeMulticlassClassifier._from_payload(json.loads(PAYLOAD_LITERAL))
classes_ = _MODEL.classes_

def predict(X):
    return _MODEL.predict(X)

def predict_proba(X):
    return _MODEL.predict_proba(X)
'''.replace('PAYLOAD_LITERAL', repr(text))
        atomic_text(path, _bundle_runtime(inference_only=True, task='multiclass') + '\n# MORPHTREE_BUNDLE_END\n' + suffix)
        return
    if getattr(model, 'task', None) == 'regression':
        text = json.dumps(model._model_payload(include_report=False), sort_keys=True, allow_nan=False)
        suffix = '''
_MODEL = MorphTreeRegressor._from_payload(json.loads(PAYLOAD_LITERAL))

def predict(X):
    return _MODEL.predict(X)
'''.replace('PAYLOAD_LITERAL', repr(text))
        atomic_text(path, _bundle_runtime(inference_only=True, task='regression') + '\n# MORPHTREE_BUNDLE_END\n' + suffix)
        return
    from .persistence import species_payload
    payload = {"format": "MorphTreeModel", "spec_version": SPEC_VERSION, "producer_version": VERSION,
               "algorithm": species_payload(model.genome_, model.params_), "n_features": model.n_features_in_,
               "feature_names": model.feature_names_in_, "booster": model.model_.to_dict(), "report": {}}
    payload['algorithm']['metadata'] = {}
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    suffix = '''
_MODEL = MorphTreeClassifier._from_payload(json.loads(PAYLOAD_LITERAL))

def predict_proba(X):
    return _MODEL.predict_proba(X)

def predict(X):
    return _MODEL.predict(X)
'''.replace("PAYLOAD_LITERAL", repr(text))
    atomic_text(path, _bundle_runtime(inference_only=True) + "\n# MORPHTREE_BUNDLE_END\n" + suffix)


def export_bundle(model, path):
    """Publish a new directory atomically; never overwrite an existing bundle."""
    import tempfile
    import shutil
    import os
    from .persistence import write_json
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix='.morphtree-', dir=destination.parent))
    try:
        model.save_model(temporary / 'model.mtree')
        regression = getattr(model, 'task', None) == 'regression'
        multiclass = getattr(model, 'task', None) == 'multiclass'
        if not regression and not multiclass:
            model.save_algorithm(temporary / 'species.morph')
        model.save_report(temporary / 'report.json')
        model.export_python(temporary / 'predictor.py')
        write_json(temporary / 'schema.json', {'names': model.feature_names_in_, 'count': model.n_features_in_})
        if multiclass:
            write_json(temporary / 'classes.json', list(model.classes_))
            atomic_text(temporary / 'README.md', '# MorphTree multiclass inference bundle\n\n'
                        'Run trusted predictor.py with Python >=3.10; no packages required.\n'
                        'Import predict or predict_proba. Probability columns follow classes_ and classes.json.\n'
                        'Numeric rows use schema order; missing values are None or NaN.\n'
                        'model.mtree is bounded declarative JSON. No rows or species recipe are bundled.\n'
                        'Report metrics are selection diagnostics. Scripts are executable; integrity is not authenticity.\n')
        elif regression:
            atomic_text(temporary / 'README.md', '# MorphTree regression inference bundle\n\n'
                        'Run trusted predictor.py with Python >=3.10. No packages required.\n'
                        'Import predict; returns continuous values, not class probabilities.\n'
                        'Numeric rows use schema order; missing values are None or NaN.\n'
                        'Named dataframe columns are aligned; unexpected columns are rejected.\n'
                        'model.mtree is bounded declarative JSON. No training rows or species recipe are bundled.\n'
                        'report.json contains training/validation selection diagnostics before optional refit.\n'
                        'Scripts are executable code; integrity is not authenticity.\n')
        else:
            atomic_text(temporary / 'README.md', '# MorphTree inference bundle\n\nRun trusted predictor.py with Python >=3.10. No packages required.\n'
                    'Import predict or predict_proba; pass numeric rows in schema order. Missing: None or NaN.\n'
                    'Named dataframe columns are aligned; unexpected columns are rejected.\n'
                    'model.mtree is validated declarative JSON. species.morph is reusable structure, not weights.\n'
                    'report.json metrics precede refit and are selection metrics. No training rows are bundled.\n'
                        'Scripts are executable code; integrity is not authenticity.\n')
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def report_markdown(report):
    tuning = report.get("tuning", {})
    lines = ["# Rapporto MorphTree", "", f"Versione: {report.get('version')}. Backend: {report.get('backend')}.", "",
             f"Arresto: `{report.get('stop_reason')}`.", f"Fingerprint strutturale: `{report.get('final_fingerprint')}`.", "",
             "## Metriche di selezione", "", "Le metriche seguenti precedono il refit. SEARCH e GUARD non sono test indipendenti.", "",
             "| Fase | SEARCH logloss | GUARD logloss |", "|---|---:|---:|"]
    for phase, prefix in (("Base", "base"), ("Struttura", "structure"), ("Finale numerico", "final")):
        lines.append(f"| {phase} | {report.get(prefix + '_search_loss', 0):.8f} | {report.get(prefix + '_guard_loss', 0):.8f} |")
    lines += ["", f"Guadagno strutturale SEARCH: {report.get('structural_gain', 0):.8f}.",
              f"Guadagno numerico SEARCH: {report.get('tuning_gain', 0):.8f}.", "", "## Protocollo", "",
              f"Plateau locale verificato: {report.get('plateau_reached')}. Ambito: {report.get('plateau_scope')}.",
              f"Tuning eseguito: {tuning.get('attempted', False)}. Esito: {tuning.get('reason')}.", "",
              "## Parametri finali", "", "```json", json.dumps(report.get("final_parameters", {}), indent=2), "```", "",
              "## Mutazioni", "", "| Round | Gene | Stadio | Decisione |", "|---:|---|---|---|"]
    for item in report.get("morph_history", []):
        lines.append(f"| {item['round']} | {item['gene']} | {item['stage']} | {item['decision']} |")
    sampling = report.get("structural_sampling")
    if sampling:
        lines += ["", "## Campionamento strutturale", "",
                  f"Frazione fissa: {sampling['row_fraction']}. Righe per albero: {sampling['sampled_rows']}/{sampling['train_rows']}.",
                  f"Politica di sampling operativa: {sampling['policy_active']}."]
        excluded = report.get("coverage", {}).get("excluded_inactive", [])
        if excluded:
            lines.append(f"Mutazioni senza effetto escluse nell'ultimo vicinato: {len(excluded)}. Dettagli nel report JSON.")
    lines += ["", "Il plateau riguarda soltanto il vicinato attivo ai parametri fissati. Non certifica un limite algoritmico globale.", ""]
    return "\n".join(lines)
