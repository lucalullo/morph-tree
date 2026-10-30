"""Explicit retained eight-round profile; importing does not train.

Call fit_classifier only with your full data and a suitable local resource budget.
This is an available configuration, not a new default or an accuracy guarantee.
"""
from morphtree import MorphTreeClassifier


def classifier():
    return MorphTreeClassifier(
        random_state=42,
        capacity={"n_estimators": 8},
        n_trials=2,
        time_budget=1800,
        memory_budget=1073741824,
    )


def fit_classifier(X, y, test):
    model = classifier()
    model.fit(X, y)
    return model, model.predict(test), model.predict_proba(test)
