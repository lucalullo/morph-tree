"""Versioned research-only roundtrip qualification, not a loader or fit permit.

Only the documented native report projections may differ. Every other byte-
decoded state path is compared. A successful state comparison alone cannot PASS:
an externally pinned, pre-save behavior witness and verified installed runtime
are required. Trust comes from the caller's qualified dossier, not this report.
"""
import copy
import hashlib
import json
import math
POLICY = 'PRODUCT002_EXPLICIT_THREE_TASK_GROWTH_ENVELOPE_V3'
NUMERIC = ('base_search_loss', 'base_guard_loss', 'structure_search_loss', 'structure_guard_loss', 'final_search_loss', 'final_guard_loss', 'structural_gain', 'tuning_gain')
FIELDS = {'active_parameters': ('B', 'list', 'algorithm.active_parameters'), 'backend': ('C', 'str', 'external installed-runtime dossier; pure/python-stdlib'), 'base_genome': ('C', 'dict', 'historical starting recipe; Base-only scope'), 'base_parameters': ('C', 'dict', 'historical starting parameters; Base-only scope'), 'config': ('C', 'dict', 'external protocol/configuration, never restored wholesale'), 'coverage': ('D', 'dict', 'discovery diagnostics, not a completion receipt'), 'error_profiles': ('D', 'list', 'diagnostic history, empty in this scope'), 'feature_schema': ('B', 'dict', 'top-level n_features/feature_names'), 'final_fingerprint': ('B', 'str', 'algorithm.structural_fingerprint'), 'final_genome': ('B', 'dict', 'algorithm.genome'), 'final_parameters': ('B', 'dict', 'algorithm.parameters'), 'freeze_reason': ('D', 'str', 'diagnostic stop explanation'), 'metrics_scope': ('C', 'str', 'TRAIN-only metric provenance, not GUARD permission'), 'morph_history': ('D', 'list', 'diagnostic history, empty in this scope'), 'plateau_reached': ('D', 'bool', 'diagnostic outcome, not scientific convergence'), 'plateau_scope': ('D', 'str', 'diagnostic scope'), 'python': ('C', 'str', 'historical interpreter; current interpreter externally pinned'), 'refit_full': ('C', 'bool', 'external learning recipe'), 'spec_version': ('B', 'int', 'top-level spec_version'), 'split': ('C', 'dict', 'TRAIN metric partition provenance, not final GUARD'), 'stop_reason': ('D', 'str', 'informational only, never COMPLETE authority'), 'structural_sampling': ('C', 'dict', 'historical TRAIN sampling provenance'), 'timings_seconds': ('D', 'dict', 'operational diagnostics, not performance evidence'), 'tuning': ('C', 'dict', 'historical tuning state, disabled in this scope'), 'version': ('B', 'str', 'top-level producer_version')}

class Unknown(ValueError):
    """Known-safe equivalence policy does not cover this material."""

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()

def decode(b, expected):
    if type(b) is not bytes or not 0 < len(b) <= 32 * 1024 ** 2:
        raise ValueError('Bounded immutable bytes required')
    if hashlib.sha256(b).hexdigest() != expected:
        raise ValueError('Artifact SHA mismatch')

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out

    def bad(_):
        raise ValueError('Nonfinite JSON')
    v = json.loads(b.decode('utf8'), object_pairs_hook=pairs, parse_constant=bad)

    def bounded(x, depth=0):
        if depth > 64:
            raise ValueError('JSON depth limit')
        if isinstance(x, dict):
            for k, y in x.items():
                if type(k) is not str:
                    raise ValueError('String keys required')
                bounded(y, depth + 1)
        elif isinstance(x, list):
            for y in x:
                bounded(y, depth + 1)
        elif type(x) is float and (not math.isfinite(x)):
            raise ValueError('Nonfinite number')
    bounded(v)
    digest(v)
    return v

def differences(a, b, path=''):
    if type(a) is not type(b):
        return [{'path': path, 'before': a, 'after': b}]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = path + '/' + k.replace('~', '~0').replace('/', '~1')
            if k not in a or k not in b:
                out.append({'path': p, 'before': a.get(k, 'ABSENT'), 'after': b.get(k, 'ABSENT')})
            else:
                out.extend(differences(a[k], b[k], p))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [{'path': path, 'before': a, 'after': b}]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path + '/' + str(i))]
    return [] if a == b else [{'path': path, 'before': a, 'after': b}]

def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)

def _binary_report(r, native):
    from morphtree.public import imported_report, PUBLIC_CONFIG_FIELDS
    from morphtree.config import MorphTreeConfig
    if type(r) is not dict:
        raise ValueError('Report object required')
    unknown = set(r) - set(NUMERIC) - set(FIELDS)
    if unknown:
        raise Unknown('Unclassified binary report fields: ' + repr(sorted(unknown)))
    types = {'str': str, 'dict': dict, 'list': list, 'bool': bool, 'int': int}
    for k in NUMERIC:
        if k in r and (not _finite(r[k])):
            raise ValueError('Nonfinite/invalid metric ' + k)
    for k, (_, typ, _) in FIELDS.items():
        if k in r and type(r[k]) is not types[typ]:
            raise ValueError('Report type ' + k)
    a = native['algorithm']
    bindings = {'active_parameters': a['active_parameters'], 'final_genome': a['genome'], 'final_parameters': a['parameters'], 'final_fingerprint': a['structural_fingerprint'], 'spec_version': native['spec_version'], 'version': native['producer_version'], 'refit_full': a['learning_recipe']['refit_full'], 'feature_schema': {'count': native['n_features'], 'names': native['feature_names']}}
    for k, v in bindings.items():
        if k in r and r[k] != v:
            raise ValueError('Contradictory report/authoritative binding ' + k)
    for k in ('morph_history', 'error_profiles'):
        if k in r and r[k] != []:
            raise Unknown('Nonempty unqualified ' + k)
    for k, target in (('base_genome', 'genome'), ('base_parameters', 'parameters')):
        if k in r and r[k] != a[target]:
            raise Unknown('Non-Base historical ' + k)
    if 'backend' in r and r['backend'] != 'python-stdlib':
        raise ValueError('Backend report mismatch')
    if 'config' in r:
        c = r['config']
        if not set(c) <= set(PUBLIC_CONFIG_FIELDS):
            raise Unknown('Unexpected config fields')
        if c.get('enable_morphing', False) or c.get('enable_tuning', False) or c.get('memory_enabled', False):
            raise Unknown('Non-Base config outside policy')
        for k, v in a['learning_recipe'].items():
            if k in c and c[k] != v:
                raise ValueError('Report/learning recipe mismatch ' + k)
        if any((type(v) not in (bool, int, float, str, type(None)) for v in c.values())):
            raise ValueError('Non-scalar operational config')
        MorphTreeConfig(**c)
    schemas = {'coverage': {'local_search_complete', 'scope', 'evaluated_neighbors', 'neighborhood_size', 'reason'}, 'split': {'guard', 'kind', 'refit_rows', 'sealed', 'search', 'train'}, 'structural_sampling': {'policy_active', 'row_fraction', 'sampled_rows', 'scope', 'train_rows'}, 'timings_seconds': {'base', 'discovery', 'refit', 'total_fit', 'tuning'}, 'tuning': {'accepted', 'attempted', 'completed', 'frozen_fingerprint', 'reason', 'trials'}}
    for k, keys in schemas.items():
        if k in r and (not set(r[k]) <= keys):
            raise Unknown('Unclassified nested report ' + k)
    if r.get('coverage', {}) != {}:
        raise Unknown('Nonempty discovery coverage outside Base scope')
    if 'split' in r:
        s = r['split']
        if set(s) != schemas['split'] or any((type(s[k]) is not int or s[k] < 0 for k in ('guard', 'refit_rows', 'search', 'train'))):
            raise ValueError('Split diagnostic types')
        if s['kind'] != 'baseline_no_selection_metrics_are_training_only' or s['sealed'] != 'not accepted by fit; evaluate externally' or len({s[k] for k in ('guard', 'search', 'refit_rows', 'train')}) != 1:
            raise Unknown('Unsupported split provenance')
    for k in ('metrics_scope', 'freeze_reason', 'plateau_scope', 'stop_reason', 'python'):
        if k in r and len(r[k]) > 4096:
            raise ValueError('Bounded informational annotation')
    if 'tuning' in r:
        t = r['tuning']
        if set(t) != schemas['tuning'] or type(t['reason']) is not str or len(t['reason']) > 4096:
            raise ValueError('Exact typed inactive tuning report')
        if any((t.get(k, False) is not False for k in ('accepted', 'attempted', 'completed'))) or t.get('trials', []) != []:
            raise Unknown('Active tuning report outside policy')
        if 'frozen_fingerprint' in t and t['frozen_fingerprint'] != a['structural_fingerprint']:
            raise ValueError('Tuning fingerprint mismatch')
    if 'timings_seconds' in r and any((not _finite(v) or v < 0 for v in r['timings_seconds'].values())):
        raise ValueError('Invalid diagnostic times')
    if 'structural_sampling' in r:
        s = r['structural_sampling']
        if set(s) != schemas['structural_sampling'] or type(s['scope']) is not str or len(s['scope']) > 4096 or (not _finite(s['row_fraction'])):
            raise ValueError('Exact typed structural sampling')
        if any((type(s[k]) is not int or s[k] < 1 for k in ('sampled_rows', 'train_rows'))) or s['sampled_rows'] != s['train_rows']:
            raise ValueError('Unsampled positive integer TRAIN counts required')
        if 'split' in r and s['train_rows'] != r['split']['train']:
            raise ValueError('Inconsistent TRAIN report counts')
        if s.get('policy_active', False) is not False or s.get('row_fraction', 1) != 1:
            raise Unknown('Sampled structure report outside scope')
    return imported_report(r)

def report_projection(r, native, task):
    if task == 'binary':
        return _binary_report(r, native)
    from morphtree.public import imported_multiclass_report, imported_regression_report
    allowed = {'task', 'imported', 'train_metrics', 'validation_metrics'}
    if task == 'regression':
        allowed.add('objective')
    if type(r) is not dict:
        raise ValueError('Report object required')
    if set(r) - allowed:
        raise Unknown('Unclassified task report fields')
    if r.get('task') != task or r.get('imported') is not True:
        raise Unknown('Only normalized retained task report qualified')
    if task == 'regression' and r.get('objective') != 'squared_error':
        raise ValueError('Objective')
    metrics = {'rmse', 'mae', 'r2'} if task == 'regression' else {'logloss', 'accuracy', 'macro_f1'}
    for group in ('train_metrics', 'validation_metrics'):
        if group not in r:
            continue
        if type(r[group]) is not dict or set(r[group]) - metrics:
            raise Unknown('Unknown metric field')
        for k, v in r[group].items():
            if not _finite(v) or (k != 'r2' and v < 0) or (k in ('accuracy', 'macro_f1') and v > 1):
                raise ValueError('Metric domain')
    return (imported_regression_report if task == 'regression' else imported_multiclass_report)(r)

def canonical_state(payload):
    if type(payload) is not dict or set(payload) != {'schema', 'constructor', 'recipe', 'classes', 'report', 'native_payloads'}:
        raise ValueError('Exact M026 envelope required')
    if payload['schema'] not in ('morph026-native-estimator-v1', 'comp002-native-estimator-v1', 'comp003-native-estimator-v1', 'comp009-native-estimator-v1', 'product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1'):
        raise ValueError('Unsupported envelope')
    if type(payload['constructor']) is not dict or set(payload['constructor']) != {'task', 'families', 'morphing', 'tuning', 'preset', 'metric', 'time_budget', 'n_trials', 'n_jobs', 'random_state', 'verbose'}:
        raise Unknown('Unknown constructor schema')
    recipe = payload['recipe']
    if type(recipe) is not dict or set(recipe) != {'schema', 'task', 'family', 'structure', 'params', 'seed', 'training_identity', 'preprocessing', 'mutations'} or recipe['schema'] != {'morph026-native-estimator-v1': 'morph026-frozen-form-v1', 'comp002-native-estimator-v1': 'comp002-frozen-form-v1', 'comp009-native-estimator-v1': 'comp009-frozen-form-v1', 'comp003-native-estimator-v1': 'comp003-frozen-form-v1', 'product002-native-estimator-v1': 'product002-frozen-form-v1', 'product003-native-estimator-v1': 'product003-frozen-form-v1', 'product006-native-estimator-v1': 'product006-frozen-form-v1'}[payload['schema']]:
        raise Unknown('Unknown frozen recipe schema')
    if type(recipe['structure']) is not dict or set(recipe['structure']) != ({'genome', 'classes', 'members', 'encoder', 'lineage'} if payload['schema'] in ('comp003-native-estimator-v1', 'comp009-native-estimator-v1', 'product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1') else {'genome', 'classes', 'members', 'encoder'}) or recipe['mutations'] != []:
        raise Unknown('Unknown structural recipe')
    outer_fields = {'GUARD_used', 'SEARCH_primary', 'available_families', 'best_family', 'classes', 'cost_scope', 'descendant_cpu', 'events', 'final_model_role', 'limitations', 'metric', 'morphing', 'optional_dependencies', 'peak_RSS', 'process_cpu_seconds', 'refit_performed', 'schema', 'seed', 'selection', 'stop_reason', 'task', 'trials', 'tried', 'tuning', 'untried', 'wall_seconds', 'worker_admission', 'variants', 'exported_variant'}
    if type(payload['report']) is not dict or set(payload['report']) - outer_fields:
        raise Unknown('Unclassified outer report fields')
    task = payload['recipe']['task']
    expected = {'binary': 'MorphTreeModel', 'multiclass': 'MorphTreeMulticlassModel', 'regression': 'MorphTreeRegressionModel'}
    if task not in expected:
        raise ValueError('Task')
    p = copy.deepcopy(payload)
    if type(p['native_payloads']) is not list or not p['native_payloads']:
        raise ValueError('Native members')
    for n in p['native_payloads']:
        if n.get('format') not in ((expected[task], {'regression': 'MorphTreeRegressionGrowthV2', 'multiclass': 'MorphTreeMulticlassGrowthV2'}.get(task)) if p['schema'] in ('product002-native-estimator-v1', 'product003-native-estimator-v1', 'product006-native-estimator-v1') else (expected[task],)) or n.get('spec_version') != 1 or n.get('producer_version') != '0.6.0.dev0':
            raise ValueError('Native format/version')
        report_projection(n['report'], n, task)
        del n['report']
    return p

def _safe_native_schema(payload):
    """Use byte-exact production validators, without fitting or inference.

    Needed even when both sides contain the same unexpected field. Equality
    alone cannot establish schema validity. No quarantine caller uses this API.
    """
    from morphtree.model import MorphTreeClassifier, MorphTreeRegressor, MorphTreeMulticlassClassifier
    classes = {'binary': MorphTreeClassifier, 'regression': MorphTreeRegressor, 'multiclass': MorphTreeMulticlassClassifier}
    for native in payload['native_payloads']:
        classes[payload['recipe']['task']]._from_payload(copy.deepcopy(native))

def _behavior(w, task, classes, nrows):
    if set(w) != ({'prediction', 'probabilities'} if task != 'regression' else {'prediction'}):
        raise ValueError('Behavior witness schema')
    p = w['prediction']
    if type(p) is not list or len(p) != nrows:
        raise ValueError('Prediction alignment')
    if task == 'regression':
        if not all((_finite(v) for v in p)):
            raise ValueError('Nonfinite predictions')
        return
    probs = w['probabilities']
    if len(probs) != nrows or any((v not in classes for v in p)):
        raise ValueError('Class alignment')
    for y, row in zip(p, probs):
        if type(row) is not list or len(row) != len(classes) or (not all((_finite(v) and 0 <= v <= 1 for v in row))):
            raise ValueError('Probability domain')
        if abs(math.fsum(row) - 1) > 1e-12:
            raise ValueError('Probability normalization')
        if classes[max(range(len(row)), key=row.__getitem__)] != y:
            raise ValueError('Class decision rule')

def compare(before_bytes, after_bytes, *, trust, witness):
    """No IO, fitting, unpickling, eval or generic annotation restoration.

    trust must be made by the qualified caller from verified runtime/dossier,
    model bytes, immutable query and witness persisted before serialization.
    This API is a cooperative verifier, not protection against a malicious caller
    that can rewrite its own trust dossier or execute arbitrary Python in process.
    """
    result = {'policy': POLICY, 'status': 'INVALID', 'A': 'UNKNOWN', 'B': 'UNKNOWN', 'C': 'UNKNOWN', 'differences': [], 'accepted_differences': [], 'new_fits': 0, 'new_trials': 0}
    try:
        required = {'runtime', 'actual_runtime', 'before_sha256', 'after_sha256', 'witness_sha256', 'query_sha256', 'source_sha256', 'recipe_sha256', 'loader_sha256', 'admission'}
        if set(trust) != required:
            raise ValueError('Exact externally pinned trust schema')
        if trust['admission'] not in ('QUALIFIED_RETAINED_ZERO_FIT', 'QUALIFIED_COMP003_SOFTWARE', 'QUALIFIED_PRODUCT002_SOFTWARE', 'QUALIFIED_COMP003_LIVE_COMPONENTS'):
            raise Unknown('Quarantine or missing exact admission')
        if trust['runtime'] != trust['actual_runtime'] or trust['runtime'].get('backend') != 'pure':
            raise ValueError('Runtime identity changed')
        a = decode(before_bytes, trust['before_sha256'])
        b = decode(after_bytes, trust['after_sha256'])
        result['differences'] = differences(a, b)
        ca = canonical_state(a)
        cb = canonical_state(b)
        _safe_native_schema(a)
        _safe_native_schema(b)
        if ca != cb:
            raise ValueError('Critical predictive/identity state differs')
        if digest(a['recipe']) != trust['recipe_sha256']:
            raise ValueError('Recipe identity')
        task = a['recipe']['task']
        for i, (x, y) in enumerate(zip(a['native_payloads'], b['native_payloads'])):
            projection = report_projection(x['report'], x, task)
            report_projection(y['report'], y, task)
            if y['report'] != projection:
                raise ValueError('Restored report not exact safe projection')
            for d in differences(x['report'], y['report'], f'/native_payloads/{i}/report'):
                key = d['path'].rsplit('/', 1)[-1]
                if key not in FIELDS:
                    raise Unknown('Unclassified report difference')
                result['accepted_differences'].append({**d, 'category': FIELDS[key][0], 'reason': 'Original safe-loader allowlist deliberately drops this typed annotation', 'authority': FIELDS[key][2]})
        result.update(B='PASS', C='PASS')
        if witness is None:
            raise Unknown('No actual pre-save witness')
        if digest(witness) != trust['witness_sha256']:
            raise ValueError('Witness SHA')
        keys = {'schema', 'runtime', 'source_sha256', 'recipe_sha256', 'query', 'loader_sha256', 'before_state_sha256', 'after_state_sha256', 'before', 'after', 'events'}
        if set(witness) != keys or witness['schema'] != 'morph032-actual-behavior-v1':
            raise ValueError('Witness schema')
        for k in ('runtime', 'source_sha256', 'recipe_sha256', 'loader_sha256'):
            if witness[k] != trust[k]:
                raise ValueError('Witness binding ' + k)
        if digest(witness['query']) != trust['query_sha256']:
            raise ValueError('Query identity')
        if witness['before_state_sha256'] != digest(a) or witness['after_state_sha256'] != digest(b):
            raise ValueError('Witness state binding')
        ev = witness['events']
        if set(ev) != {'captured_before_save', 'saved', 'safe_loaded', 'captured_after_load'} or not all((type(v) is int for v in ev.values())):
            raise ValueError('Witness event schema')
        if not ev['captured_before_save'] < ev['saved'] < ev['safe_loaded'] < ev['captured_after_load']:
            raise ValueError('Witness acquired too late or invalid timeline')
        n = len(witness['query']['rows'])
        for v in ('before', 'after'):
            _behavior(witness[v], task, a['classes'], n)
        if witness['before'] != witness['after']:
            raise ValueError('Actual predictions/probabilities differ')
        result.update(status='PASS', A='PASS', reason='Exact math, actual behavior, identity and explicit safe projection')
    except Unknown as e:
        result.update(status='UNQUALIFIED', reason=str(e))
    except (KeyError, TypeError, ValueError, IndexError, RecursionError, OverflowError) as e:
        result.update(status='REJECTED', reason=type(e).__name__ + ': ' + str(e))
    return result
