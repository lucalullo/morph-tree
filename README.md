# MorphTree

[![PyPI version](https://img.shields.io/pypi/v/morphtree)](https://pypi.org/project/morphtree/)

**Adaptive tree learning for tabular data, with structural evolution and a simple Python API.**

MorphTree 2.1.0 supports binary classification, multiclass classification and single-target regression. It screens applicable tree families, selects a parent, diagnoses training errors and evaluates structural changes to the executable learning program. A descendant replaces its parent only when the selection criteria are satisfied; otherwise the parent is retained.

Automatic preprocessing, structural evolution, final numerical optimization and model persistence are available through `fit()` and `predict()`.

## Why MorphTree?

MorphTree explores changes to the tree-learning program itself, not only its
numerical parameters. A fitted model can be used directly for predictions;
inspecting or exporting the selected learning program is optional.

### Project status

An early-stage system under active development. Training, prediction,
structural search, and export are implemented, but known functional limitations
are still being addressed. A general predictive benefit from structural
evolution has not yet been established.

### When to use it

For research into evolving tree-learning rules, examining structural decisions,
or exploring reuse of a selected learning program.

### When to consider alternatives

Conventional tree ensembles are better starting points when fast training or
extensively validated predictive performance is the priority. MorphTree's
search requires additional computation and may retain its initial program if
no proposed modification meets the selection criteria.

## Install

```bash
pip install morphtree
```

Prebuilt wheels include native components on supported platforms; see the
[installation guide](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md)
for compatibility and build requirements.

## Quick start

The examples use scikit-learn datasets (`pip install scikit-learn`).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = MorphTreeClassifier(random_state=42).fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
```

The same classifier accepts multiclass targets; use `MorphTreeRegressor` for
regression. The fitted model can predict directly, with no export step.
See [examples](https://github.com/lucalullo/morph-tree/tree/main/examples/).

## Structural evolution

During `fit()`, MorphTree screens tree families, diagnoses errors, evaluates
structural descendants, retains accepted changes, and runs final Optuna tuning.
The current mutation grammar adjusts split-objective weights and tree-growth
policy; it does not generate arbitrary algorithms. Search adds computation,
but is not repeated when predicting with a saved fitted model.

`mode="baseline"` disables structural evolution; `use_optuna=False` disables
final numerical tuning. See [architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md)
and the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md).

## Inspect and reuse

```python
print(model.get_algorithm_program())
print(model.get_mutation_history())
model.save("morphtree-model.json")
restored = MorphTreeClassifier.load("morphtree-model.json")
```

Advanced APIs can export a standalone fitted predictor or the selected learning
program for training new weights without repeating structural search. Training
from an exported program requires compatible encoded data and the matching
MorphTree runtime; see the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md).

## Limitations

- Structural evolution is not guaranteed to improve predictive performance; proposed mutations may all be rejected.
- Training cost is higher than fitting a single fixed model because multiple candidates are evaluated.
- Known functional limitations are under active investigation; consult the [qualification scope](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md).
- Performance and resource requirements depend on the dataset and search budget.

See the [comparison report](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md),
[capacity study](https://github.com/lucalullo/morph-tree/blob/main/docs/CAPACITY.md),
and [benchmark methodology](https://github.com/lucalullo/morph-tree/blob/main/BENCHMARKS.md).

## Documentation

- [Installation](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md)
- [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md)
- [Architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md)
- [Data and resource limits](https://github.com/lucalullo/morph-tree/blob/main/docs/PUBLIC_API_LIMITS.md)
- [Examples](https://github.com/lucalullo/morph-tree/tree/main/examples/)
- [Qualification](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md)
- [Contributing](https://github.com/lucalullo/morph-tree/blob/main/CONTRIBUTING.md)

## License

MorphTree is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/morph-tree/blob/main/LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
