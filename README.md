# MorphTree

**Adaptive tree learning for tabular data, with structural evolution and a simple Python API.**

MorphTree 2.1.0 supports binary classification, multiclass classification and single-target regression. It screens applicable tree families, selects a parent, diagnoses training errors and evaluates structural changes to the executable learning program. A descendant replaces its parent only when the selection criteria are satisfied; otherwise the parent is retained.

Automatic preprocessing, structural evolution, final numerical optimization and model persistence are available through `fit()` and `predict()`.

Discovering a learning program involves more work than fitting a single fixed model: MorphTree compares families, diagnoses errors, evaluates descendants and tunes the retained program. This search happens during `fit()`. A saved model can then predict without repeating it, while an exported frozen program can train new weights without rerunning the search, within the advanced API's data and runtime requirements. These are different forms of reuse; their costs and predictive quality depend on the workload.

## Installation

```bash
pip install morphtree
```

This command selects the latest compatible release available on PyPI. Compatible prebuilt wheels include the native components and require no C compiler. See the [installation guide](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md) for supported environments, local wheel installation and source-build requirements.

## Quick start

These independent examples require the optional scikit-learn package: `pip install scikit-learn`.

### Binary classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MorphTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print("Test accuracy:", model.score(X_test, y_test))
```

### Multiclass classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MorphTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print("Test accuracy:", model.score(X_test, y_test))
```

### Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MorphTreeRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test R2:", model.score(X_test, y_test))
```

`predict()` returns a NumPy array of shape `(n_samples,)`. Classification `predict_proba()` returns a numeric NumPy array of shape `(n_samples, n_classes)`; column `j` follows `classes_[j]`. Probabilities are finite, lie in `[0, 1]` and sum to one within floating-point tolerance. NumPy slicing such as `probabilities[:, 1]` works directly; select the column for your intended class.

Classifier `score()` reports accuracy; regressor `score()` reports R². Each example evaluates on its held-out TEST split.

## Structural evolution

MorphTree evaluates changes to its learning program as well as numerical parameters:

```text
Preprocess data and plan resources
    -> Screen applicable tree families and select one parent
    -> Diagnose errors from out-of-fold training evidence
    -> Propose and train structural descendants
    -> Evaluate on internal SEARCH data; accept or roll back
    -> Continue from the retained parent when the budget permits
    -> Freeze the learning program
    -> Run final Optuna numerical tuning and refit
```

The current finite mutation grammar changes split-objective weighting and tree-growth policy. Mutations execute during descendant training; they are not merely annotations. Accepted descendants can become parents for subsequent proposals. Evolution never forces acceptance, and a completed fit may retain its initial parent after rejecting every proposal.

Adaptive evolution and final Optuna tuning are enabled by default. `mode="baseline"` provides a control without structural evolution; `use_optuna=False` disables the final numerical search. These options change the scope of fitting. See the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md) and [architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md).

## Data and resources

The public estimators accept supported NumPy arrays, pandas DataFrames and row-based tabular input. Preprocessing handles numerical and categorical columns, including supported missing-value representations. Preserve the training feature schema when predicting. Missing-value and category handling are learned from the permitted training partition.

Prediction uses internal batching, preserving row order. Training does not silently discard observations to pass resource admission. Memory, time, feature cardinality and model complexity still determine whether a workload can complete. See [data and resource limits](https://github.com/lucalullo/morph-tree/blob/main/docs/PUBLIC_API_LIMITS.md) for input contracts and qualified workloads.

The Python learning program remains mutable. Compiled C kernels accelerate supported numerical operations and are selected automatically; normal use requires no backend configuration. Python retains the evolutionary controller and operations outside those kernels.

## Capacity and defaults

The default constructor is the recommended starting point. The initial screening/evolution capacity is two rounds; with `capacity=None`, the final numerical search uses a resource-dependent domain. The selected final round count is therefore not necessarily two. Automatic trial count currently resolves to two Optuna trials.

An explicit eight-round profile is available through the same API:

```python
eight_round_classifier = MorphTreeClassifier(
    random_state=42, capacity={"n_estimators": 8}
)
eight_round_regressor = MorphTreeRegressor(
    random_state=42, capacity={"n_estimators": 8}
)
```

This fixes the round capacity for the profile; it is not guaranteed to exceed the final capacity selected by the automatic policy. More rounds can improve accuracy and increase work, but can also worsen generalization. The [capacity comparison](https://github.com/lucalullo/morph-tree/blob/main/docs/CAPACITY.md) reports results for 2, 4 and 8 rounds across six datasets and explains why the defaults remain unchanged.

Other controls include `max_mutations`, `n_trials`, `time_budget`, `memory_budget`, `n_jobs` and `mode`. Their semantics and resource limits are documented in the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md).

## Inspect, save and reuse

After fitting the binary classifier above:

```python
print(model.get_algorithm_program())
print(model.get_recipe())
print(model.get_mutation_history())
print(model.get_search_report())
print(model.cost_report())

model.save("morphtree-model.json")
restored = MorphTreeClassifier.load("morphtree-model.json")
restored_predictions = restored.predict(X_test)
```

`get_algorithm_program()` describes the retained family's executable learning rules. `get_recipe()` also records its numerical parameters, fitted preprocessing and class ordering; inspecting a recipe does not train a model. `save()` and `load()` preserve the learned weights for inference on new observations with the same feature schema, without repeating discovery. Regression uses the same `save()` method and `MorphTreeRegressor.load()`.

```python
model.export_standalone("predictor.py", output_format="numpy")
```

The standalone predictor includes the fitted weights and performs inference without importing MorphTree; this explicit NumPy format requires NumPy. The default export format remains dependency-free Python lists.

To export the learning program for a separate training operation:

```python
model.export_algorithm("training_program.py")
```

Exporting does not train. The generated `train_encoded()` function trains new weights for the frozen family, rules and numerical parameters, without repeating family screening, OOF diagnosis, mutations or Optuna. It requires the exact MorphTree runtime used for export, input encoded with the exported preprocessing, compatible task/class labels in compact numeric buffers and an explicit TRAIN identity. It is an advanced training fragment, not a raw-data `fit(X, y)` replacement or a standalone predictor. The [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md) documents its encoded-data example and integrity requirements, alongside reports, genealogy and model variants.

Calling the ordinary estimator's `fit()` again runs its configured learning process; loading a model does not turn `fit()` into frozen-program training. Evaluate a reused program on fresh held-out data before relying on it for a new problem: avoiding another search does not guarantee better accuracy or a particular speedup.

## Supported environments

The verified runtime scope is CPython 3.12–3.14 on Windows x86-64 and Ubuntu 24.04 WSL2 x86-64. Linux wheel ABI auditing is separate from full runtime qualification on other distributions. Python 3.15, macOS and other architectures are outside the established scope.

Optuna and NumPy are core dependencies. pandas and scikit-learn are optional integrations. NumPy predictions support standard slicing, vectorized operations and compatible scikit-learn metrics; this does not imply support for every scikit-learn estimator protocol.

## Evidence and limitations

Predictive quality, training cost and memory use depend on the dataset, family, capacity and search budget. Local checks cover native dispatch, the three tasks, persistence, exported inference and structural decision handling within the [qualification scope](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md).

- Accepted mutations do not by themselves prove better generalization; controlled comparisons are needed to isolate their contribution.
- An independent predictive benefit from structural evolution has not been established across diverse real-world benchmarks.
- Screening, OOF diagnostics, descendants and Optuna add work beyond a single fixed model fit.
- Kernel acceleration does not imply an equivalent speedup for the complete pipeline.
- Capacity comparisons show dataset-specific gains and regressions, not a universally optimal setting.

Measured results and their limitations are recorded in the [comparison report](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md), [capacity study](https://github.com/lucalullo/morph-tree/blob/main/docs/CAPACITY.md) and [benchmark methodology](https://github.com/lucalullo/morph-tree/blob/main/BENCHMARKS.md).

## Documentation

- [Installation](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md)
- [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md)
- [Architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md)
- [Examples](https://github.com/lucalullo/morph-tree/tree/main/examples/)
- [Qualification](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md)
- [Contributing](https://github.com/lucalullo/morph-tree/blob/main/CONTRIBUTING.md)

## License

MorphTree is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/morph-tree/blob/main/LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
