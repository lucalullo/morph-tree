# MorphTree

**Adaptive tree learning for tabular data, with structural evolution and a simple Python API.**

MorphTree is a machine learning library for binary classification, multiclass classification, and single-target regression. It evaluates supported tree-learning strategies, analyzes training errors, and explores structural changes to its learning program. Proposed changes are retained only when they satisfy the search evaluation criteria.

MorphTree combines automatic tabular preprocessing, evolutionary search, optional numerical optimization, and model persistence behind a familiar `fit()` / `predict()` interface.

## Install

MorphTree **2.1.0** is available from the project source on GitHub. To install this version directly:

```bash
python -m pip install "git+https://github.com/lucalullo/morph-tree.git@main"
```

Verify the installed package:

```python
import morphtree
print(morphtree.__version__)
```

The updated 2.1.0 native distribution has **not yet been published to PyPI**. An installation from the package index is not necessarily the same build as the current GitHub source. Installing from source requires a compatible C toolchain; compatible prebuilt wheels do not require a compiler. See [installation details](docs/INSTALLATION.md).

## Quick start

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

The mutation grammar is intentionally finite. MorphTree does not autonomously invent unrestricted Python algorithms or guarantee that evolution improves predictive accuracy. See the [architecture](docs/ARCHITECTURE.md) and [research record](research/README.md).

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

The eight-round configuration is **optional**, not the default. More capacity can increase training cost and does not guarantee better generalization. Advanced controls include `max_mutations`, `n_trials`, `time_budget`, `memory_budget`, `n_jobs`, and `mode`. See the [API reference](docs/API.md) for details.

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

Standalone inference does not rerun evolutionary training. Training-program export is a separate advanced facility documented in the [API reference](docs/API.md). Use compatible package versions when loading saved models; see [migration and persistence guidance](docs/MIGRATION.md).

## Environments, development, and dependencies

- **Python:** CPython 3.12-3.14 on the qualified x86-64 Windows and Linux profiles.
- **Windows:** native wheel builds and installed-package checks have been exercised with supported Python versions.
- **Linux:** local Ubuntu/WSL2 qualification and hosted Kaggle smoke tests have exercised the native backend.
- **macOS and other architectures:** not part of the documented qualification scope for this release.
- **Dependencies:** Optuna is part of the core workflow; pandas and scikit-learn are optional integrations.

The public classifier and regressor support the scalable training path, including internal batching for prediction. Dataset admission and completion still depend on available memory, execution budgets, feature characteristics, and workload size. Rows are not silently discarded to satisfy an admission limit.

For supported environments, build instructions, and resource considerations, see [installation](docs/INSTALLATION.md), [public API limits](docs/PUBLIC_API_LIMITS.md), and [qualification scope](docs/QUALIFICATION.md).

## Architecture and performance

The learning logic, search controller, and public interface are implemented in Python. Native C kernels handle selected histogram and tree-traversal operations. Users do not need to interact with the C backend when fitting or predicting with the public API.

The compiled kernels are not a separate learning algorithm, and they do not accelerate every step of evolutionary search. End-to-end runtime can be dominated by tree-family screening, out-of-fold diagnostics, mutation trials, and Optuna optimization.

Performance and predictive accuracy depend on the dataset, hardware, search settings, and chosen capacity. Measurements are documented in the [benchmark methodology](BENCHMARKS.md) and [comparison report](docs/COMPARISON.md); they do not establish general superiority over other tabular learners. In particular, improvements from additional tree capacity must not be attributed to structural evolution without an equal-capacity comparison.

## Documentation

- [API reference](docs/API.md)
- [Installation guide](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Examples](examples/)
- [Performance comparisons](docs/COMPARISON.md)
- [Qualification and supported scope](docs/QUALIFICATION.md)
- [Resource limits](docs/PUBLIC_API_LIMITS.md)
- [Research record](research/README.md)
- [Contributing](CONTRIBUTING.md)
- [Release and publishing guide](PUBLISHING.md)

## Limitations

- Structural mutations are evaluated conditionally and may all be rejected.
- An evolutionary improvement in predictive generalization has not been independently established across diverse real-world benchmarks.
- Final tuning and evolutionary evaluation can make fitting substantially more expensive than a single fixed-model fit.
- Higher tree capacity is opt-in and can increase runtime and resource use.
- Native kernels accelerate selected operations, not the complete training pipeline; some paths can consume more memory.
- Large-scale training, cross-platform portability, and competitive predictive performance require additional independent validation.

## Status

**MorphTree 2.1.0** is the current GitHub source version. Its Python+C implementation has passed scoped local technical checks, a manual hosted CI build/test workflow, and initial Kaggle smoke tests for binary classification, multiclass classification, and regression. These checks include use of the native kernels and model save/load.

The updated 2.1.0 native build has **not yet been published to TestPyPI or PyPI**. Large-dataset Kaggle benchmarking and final public-API verification remain separate from the completed smoke tests. The [qualification record](docs/QUALIFICATION.md) describes the verified scope; source availability is not a claim of universal compatibility or accuracy.

## License

MorphTree is created by Luca Lullo and released under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
