"""Native candidate workflow. Importing does not train; callers provide the full data.

Install a compatible verified native wheel first. See docs/QUALIFICATION.md.
The returned model retains evolution, final default Optuna, and full-input refit.
"""
from morphtree import MorphTreeClassifier, MorphTreeRegressor


def train_classifier(X, y, test):
    model = MorphTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, model.predict(test)


def train_regressor(X, y, test):
    model = MorphTreeRegressor(random_state=42)
    model.fit(X, y)
    return model, model.predict(test)


def save_and_reload(model, path):
    """Keep every genuine retained variant and verify the saved file on loading."""
    checksum = model.save(path)
    return type(model).load(path, sha256=checksum)


def inspect_model(model):
    return {
        'program': model.get_algorithm_program(),
        'recipe': model.get_recipe(),
        'genealogy': model.get_mutation_history(),
        'variants': model.available_variants(),
        'cost': model.cost_report(),
    }
