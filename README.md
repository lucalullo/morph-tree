# MorphTree

**Adaptive tree learning for tabular data, with structural evolution and a simple Python API.**

MorphTree is a machine learning library for binary classification, multiclass classification, and single-target regression. It evaluates supported tree-learning strategies, analyzes training errors, and explores structural changes to its learning program. Proposed changes are retained only when they satisfy the search evaluation criteria.

MorphTree combines automatic tabular preprocessing, evolutionary search, optional numerical optimization, and model persistence behind a familiar `fit()` / `predict()` interface.

## Install

For the MorphTree **2.1.0** distribution on PyPI, where a compatible artifact is available:

```bash
python -m pip install "morphtree==2.1.0"
```

To install the development source from GitHub instead:

```bash
python -m pip install "git+https://github.com/lucalullo/morph-tree.git@main"
```

The `main` branch can change independently of a released package. Check the installed version with `python -c "import morphtree; print(morphtree.__version__)"`. Building from source requires a suitable C compiler; installing a compatible prebuilt wheel does not. See the [installation guide](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md).

## Quick start

The examples below use the optional scikit-learn and pandas packages to load datasets. Install them separately if needed (`python -m pip install scikit-learn pandas`).

Binary classification with a real dataset:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from morphtree import MorphTreeClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MorphTreeClassifier(random_state=42)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
print("Test accuracy:", model.score(X_test, y_test))
```

The example uses scikit-learn to load and split the dataset; scikit-learn is not required by MorphTree's core training API. Evaluate predictive quality on data not used for fitting.

Multiclass classification uses the same estimator:

```python
from sklearn.datasets import load_iris
from morphtree import MorphTreeClassifier

X, y = load_iris(return_X_y=True, as_frame=True)
model = MorphTreeClassifier(random_state=42).fit(X, y)
labels = model.predict(X)
probabilities = model.predict_proba(X)
print(model.classes_)
```

Regression uses `MorphTreeRegressor`:

```python
from sklearn.datasets import load_diabetes
from morphtree import MorphTreeRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
model = MorphTreeRegressor(random_state=42).fit(X, y)
predictions = model.predict(X)
```

These two short examples demonstrate the API, not a measured generalization score. Use a held-out validation or test split for actual evaluation.

## Highlights

- **Automatic task handling:** one classifier for binary and multiclass targets, plus a dedicated regressor.
- **Tabular preprocessing:** supported numerical, categorical, and missing values are handled inside the estimator.
- **Tree-family screening:** MorphTree evaluates applicable built-in learners and selects a starting parent.
- **Error-guided search:** out-of-fold training diagnostics help identify regions where the model struggles.
- **Structural evolution:** changes to the executable learning program are tested, accepted, or rolled back according to evaluation criteria.
- **Multi-generation search:** accepted descendants can become parents of subsequent proposals; acceptance is never forced.
- **Final optimization:** Optuna tunes numerical parameters after the selected program is frozen, by default.
- **Model inspection and persistence:** inspect the program and mutation history, save and reload fitted models, and export standalone inference.
- **Native execution:** compiled C kernels accelerate selected training and prediction operations while the public API and evolutionary controller remain in Python.

## How MorphTree works

MorphTree does more than tune the numerical parameters of a fixed tree model. Its adaptive training process evaluates learning strategies and tests bounded structural changes to the learning program:

```text
Input data (X, y)
      |
      v
Preprocessing and resource planning
      |
      v
Screen supported tree families and select a parent
      |
      v
Diagnose training errors using out-of-fold evidence
      |
      v
Propose and evaluate structural mutations
      |
      +--> Accept improvement or roll back
      |
      v
Freeze the selected learning program
      |
      v
Optional Optuna tuning and final refit
      |
      v
Predict, inspect, save, or export
```

Evolution is enabled in the normal adaptive mode, but a successful mutation is **not guaranteed**. A model may complete training with every structural proposal rejected. `mode="baseline"` is available for controlled comparisons without structural evolution.

The mutation grammar is intentionally finite. MorphTree does not autonomously invent unrestricted Python algorithms or guarantee that evolution improves predictive accuracy. See the [architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md) and [research record](https://github.com/lucalullo/morph-tree/blob/main/research/README.md).

## Configuration and inspection

The default constructor is the intended starting point:

```python
model = MorphTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

Adaptive search and final Optuna tuning are enabled by default, so a full fit may take longer than training a single fixed tree model. To skip final numerical tuning:

```python
model = MorphTreeClassifier(random_state=42, use_optuna=False)
```

For an explicitly higher-capacity configuration:

```python
model = MorphTreeClassifier(random_state=42, capacity={"n_estimators": 8})
```

The eight-round configuration is **optional**, not the default. More capacity can increase training cost and does not guarantee better generalization. Advanced controls include `max_mutations`, `n_trials`, `time_budget`, `memory_budget`, `n_jobs`, and `mode`. See the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md) for details.

After fitting, you can inspect the selected program and the search decisions:

```python
print(model.get_algorithm_program())
print(model.get_mutation_history())
print(model.get_search_report())
print(model.cost_report())
```

Mutation history distinguishes accepted and rejected changes. Inspection reports describe the search actually performed, not an assumed improvement.

## Persistence and export

```python
model.save("morphtree-model.json")
restored = MorphTreeClassifier.load("morphtree-model.json")
predictions = restored.predict(X_test)
```

Fitted models can also export a standalone inference program:

```python
model.export_standalone("predictor.py")
```

Standalone inference does not rerun evolutionary training. Training-program export is a separate advanced facility documented in the [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md). Use compatible package versions when loading saved models; see [migration and persistence guidance](https://github.com/lucalullo/morph-tree/blob/main/docs/MIGRATION.md).

## Environments, development, and dependencies

- **Python:** CPython 3.12-3.14 on the qualified x86-64 Windows and Linux profiles.
- **Windows:** native wheel builds and installed-package checks have been exercised with supported Python versions.
- **Linux:** local Ubuntu/WSL2 qualification and hosted Kaggle smoke tests have exercised the native backend.
- **macOS and other architectures:** not part of the documented qualification scope for this release.
- **Dependencies:** Optuna is part of the core workflow; pandas and scikit-learn are optional integrations.

The public classifier and regressor support the scalable training path, including internal batching for prediction. Dataset admission and completion still depend on available memory, execution budgets, feature characteristics, and workload size. Rows are not silently discarded to satisfy an admission limit.

For supported environments, build instructions, and resource considerations, see [installation](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md), [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md), and [qualification scope](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md).

## Architecture and performance

The learning logic, search controller, and public interface are implemented in Python. Native C kernels handle selected histogram and tree-traversal operations. Users do not need to interact with the C backend when fitting or predicting with the public API.

The compiled kernels are not a separate learning algorithm, and they do not accelerate every step of evolutionary search. End-to-end runtime can be dominated by tree-family screening, out-of-fold diagnostics, mutation trials, and Optuna optimization.

Performance and predictive accuracy depend on the dataset, hardware, search settings, and chosen capacity. Measurements are documented in the [benchmark methodology](https://github.com/lucalullo/morph-tree/blob/main/BENCHMARKS.md) and [comparison report](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md); they do not establish general superiority over other tabular learners. In particular, improvements from additional tree capacity must not be attributed to structural evolution without an equal-capacity comparison.

## Documentation

- [API reference](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md)
- [Installation guide](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md)
- [Architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md)
- [Examples](https://github.com/lucalullo/morph-tree/tree/main/examples/)
- [Performance comparisons](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md)
- [Qualification and supported scope](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md)
- [Resource considerations](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md)
- [Research record](https://github.com/lucalullo/morph-tree/blob/main/research/README.md)
- [Contributing](https://github.com/lucalullo/morph-tree/blob/main/CONTRIBUTING.md)
- [Release and publishing guide](https://github.com/lucalullo/morph-tree/blob/main/PUBLISHING.md)

## Limitations

- Structural mutations are evaluated conditionally and may all be rejected.
- An evolutionary improvement in predictive generalization has not been independently established across diverse real-world benchmarks.
- Final tuning and evolutionary evaluation can make fitting substantially more expensive than a single fixed-model fit.
- Higher tree capacity is opt-in and can increase runtime and resource use.
- Native kernels accelerate selected operations, not the complete training pipeline; some paths can consume more memory.
- Large-scale training, cross-platform portability, and competitive predictive performance require additional independent validation.

## Release scope

This README describes **MorphTree 2.1.0**. Technical checks have covered native training, classification, regression, model persistence, and installed-package workflows on the documented environments, including a manual hosted CI run and Kaggle smoke tests. These checks do not establish universal platform compatibility, predictive superiority, or performance on arbitrary large datasets.

See the [qualification record](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md) for the verified scope and the [publishing guide](https://github.com/lucalullo/morph-tree/blob/main/PUBLISHING.md) for release procedures.

## License

MorphTree is created by Luca Lullo and released under the [MIT License](https://github.com/lucalullo/morph-tree/blob/main/LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
