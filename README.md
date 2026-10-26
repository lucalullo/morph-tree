# MorphTree

**Native tree learning with introspection and structural evolution.**

MorphTree is an experimental Python library for **binary classification, multiclass classification and regression** on tabular data. Instead of stopping at the selection of a fixed learner, it compares its qualified native tree-learning families, selects a single parent and examines where that parent makes errors. It then evaluates executable changes to the parent's learning rules, retains improvements that meet its validation criteria and rolls back changes that do not.

The evolutionary search runs within a finite budget. Once it stops, MorphTree freezes the selected learning program and, by default, uses **Optuna for final numerical tuning**. The result is a trained model ready for prediction, accompanied by its program, recipe, mutation history and search report.

The proprietary learning engine is written entirely in **Python**, without compiled extensions of its own. Third-party dependencies may contain compiled components. Structural evolution is implemented and tested, but independent predictive gains from mutation have not yet been established.

## Highlights

- **Two explicit estimators:** `MorphTreeClassifier` for binary and multiclass classification, and `MorphTreeRegressor` for regression.
- **Automatic workflow:** `fit(X, y)` handles supported input data, initial family screening, parent selection, diagnosis, structural search and model finalization.
- **Introspection-guided evolution:** candidate mutations change executable learning rules, not just numerical hyperparameters.
- **Verified rollback:** non-improving descendants are rejected and the current parent is preserved.
- **Optuna by default:** numerical tuning runs after the structural program is frozen; `use_optuna=False` disables only that final tuning.
- **Native tabular preprocessing:** numeric and categorical features, missing values and supported pandas nullable types.
- **Portable model artifacts:** bounded JSON save/load, algorithm inspection and standalone inference export.

## Installation

MorphTree requires **Python 3.12–3.14**. The qualified release environments are Windows AMD64 with Python 3.12–3.14 and Ubuntu 24.04 WSL2 x86_64 with Python 3.12. Other platform/version combinations have not been fully qualified. Python 3.15 remains experimental and is excluded from the supported installation range.

Install the published package with:

```bash
python -m pip install morphtree
```

Or install from the repository root:

```bash
python -m pip install .
```

Optuna is installed as a required dependency. See [installation details](docs/INSTALLATION.md) and [dependency information](DEPENDENCIES.md).

## Quick start

### Classification

Use the same estimator for binary and multiclass targets:

```python
from morphtree import MorphTreeClassifier

model = MorphTreeClassifier(random_state=42, verbosity=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Regression

```python
from morphtree import MorphTreeRegressor

model = MorphTreeRegressor(random_state=42, verbosity=0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The estimator determines the task. No `task` parameter or manual learner selection is needed. Binary versus multiclass classification is handled inside `MorphTreeClassifier`.

See the [binary](examples/binary.py), [multiclass](examples/multiclass.py) and [regression](examples/regression.py) examples.

## How MorphTree works

```text
Tabular dataset (X, y)
          |
          v
Screen qualified, applicable native tree learners
          |
          v
Choose one parent and diagnose its errors
          |
          v
Propose an executable structural mutation
          |
          v
Train and evaluate the descendant
          |
          +-- Verified improvement --> Accept as new parent --+
          |                                                   |
          +-- Otherwise -----------> Roll back ---------------+
          |                                                   |
          +<------------- Repeat within budget <--------------+
          |
          v
Freeze the selected learning program
          |
          v
Final numerical Optuna tuning (enabled by default)
          |
          v
Trained model + program + recipe + mutation history
```

MorphTree currently qualifies a limited set of native learning families and a **depthwise structural mutation**. A run can correctly reject every proposed descendant and return the original parent. This is not a failure of the evolutionary process. The implementation does not claim that all envisioned mutation operators or native tree families are already available.

For more detail, see the [architecture](docs/ARCHITECTURE_1_0.md) and [algorithm specification](docs/Relazione_MorphTree.txt).

## Inspecting and saving models

After fitting, you can inspect what MorphTree actually did and save the trained result:

```python
from hashlib import sha256
from pathlib import Path
from morphtree import MorphTreeClassifier

# model is an already fitted MorphTreeClassifier.
program = model.get_algorithm_program()
recipe = model.get_recipe()
history = model.get_mutation_history()
report = model.get_search_report()

path = Path("model.json")
model.save(path)
checksum = sha256(path.read_bytes()).hexdigest()
restored = MorphTreeClassifier.load(path, sha256=checksum)
predictions = restored.predict(X_test)

restored.export_standalone("predictor.py")
```

The JSON model retains learned state and preprocessing and can be loaded without retraining. For a transferred model, obtain the expected SHA-256 digest from a trusted source; calculating a digest from an untrusted file alone does not authenticate it. `export_standalone()` exports **inference**, not a standalone implementation of the evolutionary training process.

You can inspect the numeric-tuning outcome through `model.report()`. Optuna is enabled by default but may stop or be skipped when a documented budget condition applies. Set `use_optuna=False` to disable final numeric tuning while keeping structural evolution enabled.

## Data handling and limits

MorphTree accepts supported dense NumPy arrays, pandas DataFrames and rectangular Python sequences. It handles numerical values, booleans, string categories, `None`, `NaN` and `pandas.NA`, including supported nullable pandas dtypes. Categorical vocabularies are learned from training data, and unseen categories are handled through dedicated encoding paths.

Current limits include **64 observed categories per categorical column**, **128 encoded features**, an approximately **2 MiB public input-payload limit** and **65,536 training rows**. Sparse data, raw datetime features and multioutput targets are not supported. Time and memory safeguards are bounded or cooperative and do not guarantee an absolute peak memory limit.

See the [API reference](docs/API_2_0.md) for input contracts, parameters, budget settings and error handling.

## Validation and performance

The native evolutionary pipeline has been exercised through real training for binary classification, multiclass classification and regression in the documented Windows and Ubuntu WSL2 profiles. These tests cover diagnosis, executed structural descendants, rollback, frozen programs, Optuna, persistence and inference. Accepted-child continuation has software-test coverage, but beneficial mutation acceptance has not yet been demonstrated in independent LIVE evaluations.

MorphTree does **not** claim universal predictive superiority, proven gains from autonomous evolution or performance leadership over established tree-learning libraries. Fit cost includes screening, out-of-fold diagnosis, descendants and final numerical tuning.

See [benchmark disclosures](BENCHMARKS.md), [qualification](docs/RELEASE_QUALIFICATION.md) and the [roadmap](ROADMAP.md).

## Documentation

- [API reference](docs/API_2_0.md)
- [Installation and dependencies](docs/INSTALLATION.md)
- [Architecture](docs/ARCHITECTURE_1_0.md)
- [Algorithm specification](docs/Relazione_MorphTree.txt)
- [Research overview](research/README.md)
- [Release qualification](docs/RELEASE_QUALIFICATION.md)
- [Contributing](CONTRIBUTING.md) and [security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## License

MorphTree is created by Luca Lullo and released under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
