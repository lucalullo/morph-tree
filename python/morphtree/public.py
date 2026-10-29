"""Public artifact boundary. Runtime connection settings are never serializable."""
from dataclasses import asdict
import math
from .config import MorphTreeConfig, canonical_hash

PUBLIC_CONFIG_FIELDS = (
    'seed', 'search_fraction', 'guard_fraction', 'enable_morphing', 'enable_tuning',
    'structural_rounds', 'structural_mutants', 'structural_patience', 'structural_evaluations',
    'search_timeout', 'freeze_on_budget', 'diagnostic_guidance', 'memory_enabled',
    'min_morph_improvement', 'guard_tolerance', 'global_guard_tolerance', 'complexity_penalty',
    'confidence_z', 'probe_fraction', 'probe_margin', 'tuner', 'tune_trials', 'tune_patience',
    'tune_timeout', 'early_stopping_rounds', 'refit_full', 'optuna_startup_trials')
RECIPE_FIELDS = ('seed', 'memory_enabled', 'refit_full', 'search_fraction', 'guard_fraction')

def public_config(config):
    return {name: getattr(config, name) for name in PUBLIC_CONFIG_FIELDS if hasattr(config, name)}

def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != set(RECIPE_FIELDS) | {'partition'}:
        raise ValueError('Invalid learning recipe fields')
    config = MorphTreeConfig(**{name: recipe[name] for name in RECIPE_FIELDS})
    if recipe['partition'] not in ('automatic', 'explicit', 'all'):
        raise ValueError('Invalid learning partition')
    return {**{name: getattr(config, name) for name in RECIPE_FIELDS}, 'partition': recipe['partition']}

def recipe_settings(recipe, config=None):
    if recipe is None:
        return config or MorphTreeConfig()
    recipe = validate_recipe(recipe)
    if config is not None:
        if config.memory_enabled != recipe['memory_enabled']:
            raise ValueError('config.memory_enabled conflicts with the frozen learning recipe')
        return config
    return MorphTreeConfig(**{name: recipe[name] for name in RECIPE_FIELDS})

def learning_fingerprint(genome, params, recipe):
    return canonical_hash({'genome': genome.to_dict(), 'parameters': asdict(params),
                           'learning_recipe': validate_recipe(recipe), 'semantics': 2})

def imported_report(data):
    """Imported annotations are untrusted. Preserve only typed, known numeric metrics.

    Operational histories/free text are deliberately not carried across artifact boundaries.
    Predictions and the explicit learning recipe are independent of this diagnostic subset.
    """
    result = {}
    for name in ('base_search_loss', 'base_guard_loss', 'structure_search_loss', 'structure_guard_loss',
                 'final_search_loss', 'final_guard_loss', 'structural_gain', 'tuning_gain'):
        value = data.get(name)
        if type(value) in (int, float):
            try:
                normalized = float(value)
            except OverflowError:
                continue
            if math.isfinite(normalized):
                result[name] = normalized
    return result

def public_metadata(metadata):
    result = {}
    if 'learning_recipe' in metadata:
        result['learning_recipe'] = validate_recipe(metadata['learning_recipe'])
    if isinstance(metadata.get('discovery_report'), dict):
        result['discovery_report'] = imported_report(metadata['discovery_report'])
    return result


def imported_regression_report(data):
    """Regression diagnostics only; never trust free-form operational annotations."""
    result = {'task': 'regression', 'objective': 'squared_error', 'imported': True}
    for group in ('train_metrics', 'validation_metrics'):
        values = data.get(group)
        if not isinstance(values, dict):
            continue
        kept = {}
        for name in ('rmse', 'mae', 'r2'):
            value = values.get(name)
            if type(value) in (int, float):
                try:
                    number = float(value)
                except OverflowError:
                    continue
                if math.isfinite(number) and (name == 'r2' or number >= 0):
                    kept[name] = number
        result[group] = kept
    return result


def imported_multiclass_report(data):
    result = {'task': 'multiclass', 'imported': True}
    for group in ('train_metrics', 'validation_metrics'):
        values = data.get(group)
        if not isinstance(values, dict):
            continue
        kept = {}
        for name in ('logloss', 'accuracy', 'macro_f1'):
            value = values.get(name)
            if type(value) in (int, float):
                try:
                    number = float(value)
                except OverflowError:
                    continue
                if math.isfinite(number) and 0 <= number and (name == 'logloss' or number <= 1):
                    kept[name] = number
        result[group] = kept
    return result
