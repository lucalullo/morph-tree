# MorphTree

**Adaptive tree learning for tabular data, with structural evolution and a simple Python API.**

MorphTree 2.1.0 supports binary classification, multiclass classification and single-target regression. It screens applicable tree families, selects a parent, diagnoses training errors and evaluates structural changes to the executable learning program. A descendant replaces its parent only when the selection criteria are satisfied; otherwise the parent is retained.

Automatic preprocessing, structural evolution, final numerical optimization and model persistence are available through `fit()` and `predict()`.

## Installation

```bash
pip install morphtree
```

This command selects the latest compatible release available on PyPI. Compatible prebuilt wheels include the native components and require no C compiler. See the [installation guide](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md) for supported environments, local wheel installation and source-build requirements.

## Quick start

The examples below use optional sample-data dependencies: `pip install scikit-learn pandas`.

### Binary classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classifier = MorphTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)
second_class_probabilities = probabilities[:, 1]
print("Probability columns:", classifier.classes_)
print("Test accuracy:", classifier.score(X_test, y_test))
```

`predict()` returns a NumPy array of shape `(n_samples,)`. Classification probabilities are a numeric NumPy array of shape `(n_samples, n_classes)`: column `j` corresponds to `classes_[j]`. Values are finite, lie in `[0, 1]` and sum to one per row within floating-point tolerance. Select the column corresponding to your intended positive class; column 1 is the second class in `classes_`.

### Multiclass classification

```python
from sklearn.datasets import load_iris
from morphtree import MorphTreeClassifier

X_multi, y_multi = load_iris(return_X_y=True, as_frame=True)
multiclass_model = MorphTreeClassifier(random_state=42).fit(X_multi, y_multi)
labels = multiclass_model.predict(X_multi)
class_probabilities = multiclass_model.predict_proba(X_multi)
print(multiclass_model.classes_)
```

### Regression

```python
from sklearn.datasets import load_diabetes
from morphtree import MorphTreeRegressor

X_reg, y_reg = load_diabetes(return_X_y=True, as_frame=True)
regressor = MorphTreeRegressor(random_state=42).fit(X_reg, y_reg)
values = regressor.predict(X_reg)
```

The multiclass and regression snippets illustrate the interface using training rows. Assess predictive quality on separate validation and test data. Classifier `score()` reports accuracy; regressor `score()` reports R².

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
print(classifier.get_algorithm_program())
print(classifier.get_mutation_history())
print(classifier.get_search_report())
print(classifier.cost_report())

classifier.save("morphtree-model.json")
restored = MorphTreeClassifier.load("morphtree-model.json")
restored_predictions = restored.predict(X_test)
```

Regression uses the same `save()` method and `MorphTreeRegressor.load()`.

```python
classifier.export_standalone("predictor.py", output_format="numpy")
```

The standalone predictor performs inference without importing MorphTree; this explicit NumPy format requires NumPy. The default export format remains dependency-free Python lists. `export_algorithm()` is a separate advanced API for training a frozen program, not a standalone inference export. Reports, recipes, genealogy, available model variants and loading-integrity checks are described in the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md).

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

## License and author

Created by [Luca Lullo](https://github.com/lucalullo). Released under the [MIT License](https://github.com/lucalullo/morph-tree/blob/main/LICENSE).
