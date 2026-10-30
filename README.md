# MorphTree

**An evolving tree learner for tabular data, with a simple Python API and compiled C acceleration.**

MorphTree screens its own supported tree-learning families, diagnoses training errors, explores executable **structural changes to its learning program**, and retains descendants only when the evaluation criteria are satisfied. After structural search, it freezes the selected program, runs final numerical tuning with Optuna by default, and refits the selected model on the eligible input data.

**Version 2.1.0.** The native implementation has passed local technical qualification within a documented scope. See [release status](#release-status) for what has **not** yet been verified or published.

## Quick start

This example uses only built-in Python lists and can be run after installing a compatible MorphTree wheel:

```python
from morphtree import MorphTreeClassifier

# Small illustrative dataset; replace it with your own X and y.
X = [[float(i % 9), float((i * 3) % 11)] for i in range(80)]
y = [int(row[0] + row[1] > 10) for row in X]

model = MorphTreeClassifier(random_state=42)
model.fit(X[:60], y[:60])  # Evolution is enabled; Optuna is on by default.

predictions = model.predict(X[60:])
probabilities = model.predict_proba(X[60:])
print("Accuracy:", model.score(X[60:], y[60:]))
print("Predictions:", predictions[:5])

model.save("morphtree-model.json")
restored = MorphTreeClassifier.load("morphtree-model.json")
assert list(restored.predict(X[60:])) == list(predictions)
```

For regression, replace `MorphTreeClassifier` with `MorphTreeRegressor` and use `predict()`; for multiclass classification, use the same classifier. `fit(X, y)` accepts supported tabular input without a separate task flag.

The public `MorphTreeClassifier` and `MorphTreeRegressor` use the scalable, columnar training path. They handle supported missing and categorical values without the legacy **2 MiB / 65,536 TRAIN rows / 128 encoded features** admission caps. Those caps remain in the separate legacy `morphtree.api.MorphTree` path; they are not limits of these public task classes. Admission depends on the estimated workspace, observed memory headroom and time budget. An admitted dataset can still exhaust resources during training; there is no promise that every dataset fits on every machine. Rows are not silently dropped, and prediction uses internal blocks without requiring manual batching. See [public API and resource limits](docs/PUBLIC_API_LIMITS.md).

## What makes MorphTree different?

1. **Native-family screening.** Evaluate the applicable built-in tree learners and choose a starting parent.
2. **Error introspection.** Use out-of-fold TRAIN diagnostics to identify areas in which the parent struggles.
3. **Structural evolution.** Propose changes to the actual learning program, train descendants, verify the results, and accept or roll back changes.
4. **Multi-generation continuation.** Subsequent proposals build on the retained parent; acceptance is never forced.
5. **Final numerical tuning.** Freeze the program before Optuna tuning (`use_optuna=True` by default), then refit the selected model.
6. **Inspectable results.** Save and load models, inspect the learned program and genealogy, export recipes and standalone inference, and access recorded costs.

The present structural mutation grammar is deliberately finite; acceptance of a new descendant is conditional, not guaranteed. MorphTree is **not** claimed to outperform other learners in general, nor has the predictive benefit of introspection-guided evolution been independently established across diverse real-world benchmarks. See the [research record](research/README.md) and [qualification scope](docs/QUALIFICATION.md).

## Python interface

```python
from morphtree import MorphTreeRegressor

model = MorphTreeRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(model.score(X_test, y_test))  # R² for regression
print(model.get_mutation_history())
print(model.get_algorithm_program())
print(model.cost_report())

model.save("regression-model.json")
```

For classification, `score()` is accuracy. Supported constructor controls include `max_mutations`, `use_optuna`, `n_trials`, `time_budget`, `mode`, `capacity`, and `n_jobs`. Supported advanced model inspection, checkpoint and export options are described in the [API guide](docs/API.md).

The normal adaptive mode keeps evolutionary search active. Explicit `mode="baseline"` is available for controlled comparisons and does **not** perform structural evolution. This distinction matters when evaluating results.

## Inspect what MorphTree learned

After `fit()`, the fitted estimator exposes the actual search record rather than a claim that every mutation succeeded:

```python
print(model.get_algorithm_program())  # Frozen learning-program description
print(model.get_mutation_history())    # Accepted and rejected structural proposals
print(model.get_search_report())       # Screening and search record
print(model.cost_report())             # Recorded costs

# Optional, with a fitted model:
model.export_standalone("predictor.py")
```

The standalone predictor exports **inference**, not a new evolutionary training run. `export_algorithm()` is a distinct, advanced training-fragment facility with additional runtime/provenance requirements; see [API documentation](docs/API.md). For binary and multiclass classification use `MorphTreeClassifier`; for single-target regression use `MorphTreeRegressor`. Unsupported inputs and resource requests fail explicitly rather than silently bypassing steps.

## Installation

**The new 2.1.0 native artifacts have not yet been published.** Installing with `pip install morphtree==2.1.0` is **not** evidence that you have this native build. For local evaluation, use the reviewed wheel supplied with the delivery, matching your Python and operating system.

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "C:\path\to\morphtree-2.1.0-py3-none-win_amd64.whl"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m morphtree --help
```

**Ubuntu WSL2:**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install /path/to/morphtree-2.1.0-py3-none-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl
.venv/bin/python -m pip check
.venv/bin/python -m morphtree --help
```

Paths are examples: replace them with the location of the **actual** wheel. Do not install both platform wheels in the same environment. Installation from source requires a suitable compiler and platform development headers; installation from a matching wheel does not.

- **Python:** CPython 3.12–3.14 on qualified 64-bit x86 systems.
- **Windows:** wheel built with MSVC, no compiler required for a compatible wheel.
- **Linux:** the wheel has auditwheel-verified manylinux ABI tags. Complete-pipeline runtime qualification remains Ubuntu 24.04 under WSL2; the ABI audit is not a claim that every Linux distribution has been tested.
- **Source installation:** requires a compatible C toolchain, a linker and system development headers; building from source compiles the native kernels.
- **Core dependency:** Optuna 5.0.0. Pandas and scikit-learn are optional integrations, not required for the basic NumPy-style workflow.

See the full [installation instructions](docs/INSTALLATION.md), [dependency record](DEPENDENCIES.md) and [security notes](SECURITY.md).

## How the engine is organized

```text
Your data (X, y)
      |
      v
Tabular preprocessing & resource planning
      |
      v
Screen tree families -> retain one parent
      |
      v
TRAIN-only diagnosis -> structural mutation -> test child
      |                                  |
      +------ accept / rollback <-------+
      |
      v
Freeze the learning program
      |
      v
Final Optuna tuning -> full-input refit -> trained model
      |
      +--> predict / predict_proba / score
      +--> save / load / report / export
```

The learning rules, evolutionary controller and public API remain in **Python**. Verified C kernels accelerate ordered histogram operations and cached tree traversal. Small prediction batches can use equivalent Python traversal to avoid native-call overhead. No third-party vendor tree learner is used as MorphTree's internal training engine.

## Verified scope and measured performance

The local evidence covers full automatic native training for binary classification, multiclass classification and regression:

| Environment | Qualified full native training |
| --- | --- |
| Windows x86-64, Python 3.12 | 10,003 × 15; 25,003 × 7; 200,003 × 15 |
| Windows x86-64, Python 3.13/3.14 | 25,003 × 7 |
| Ubuntu 24.04 WSL2 x86-64, Python 3.12–3.14 | 25,003 × 7 |

On Windows/Python 3.12, **single paired fits** with 200,003 rows and 15 columns measured these native-vs-Python speedups:

| Task | Observed whole-fit speedup |
| --- | ---: |
| Binary classification | 1.64× |
| Multiclass classification | 2.09× |
| Regression | 1.16× |

These are **individual measurements, not replicated statistical estimates**. They do not establish universal accuracy, speed or memory superiority; some native paths use more memory. Inference has been checked up to 1,000,003 rows, but **million-row training has not been qualified**. Large Linux training, Python 3.15, macOS, broad Linux runtime portability and hosted Kaggle are also unqualified. The 200,003-row results are supervised measurements of the public scalable `fit()` path, with resource accounting and stop callbacks. They are not new unsupervised replications or guarantees that default resource planning will complete any dataset of that size. Zero-fit admission checks are reported separately from completed training.

## Explicit higher capacity

```python
model = MorphTreeClassifier(random_state=42, capacity={"n_estimators": 8})
```

Eight rounds remain an available configuration, not a new default. On the frozen stellar VALIDATION, PREDICTIVE110 improved balanced accuracy from 0.879211 to 0.920422 and reached log-loss 0.144333. Equal-capacity static and evolutionary models were identical and all four structural proposals rolled back. This is a capacity result, not an established mutation benefit or confirmation on a new independent TEST. See [comparison](docs/COMPARISON.md) and `examples/capacity8.py`.

See [measurements](docs/COMPARISON.md), [qualification details](docs/QUALIFICATION.md) and [benchmark methodology](BENCHMARKS.md) for caveats and evidence provenance.

## Repository layout

```text
.github/workflows/       Manual review-controlled build and release workflows
docs/                    API, architecture, qualification, migration, handoff
examples/                Minimal task examples and notebooks
python/morphtree/        Python implementation and native C sources
python/tests/product/   Package-level tests
research/                Curated historical findings, including negative results
tests/                   Integration, packaging and metadata tests
tools/                   Build, inventory, and installed-package checks
pyproject.toml           Distribution metadata
SOURCE_MANIFEST.json     Reviewed source inventory and SHA-256 digests
```

### Building and contributing

Builds are checked against the repository source inventory and performed in a separate source copy. Building a source distribution requires a compatible system toolchain. The CI workflows are **manual** (`workflow_dispatch`); their presence is not evidence that remote CI or TestPyPI publishing has already passed.

See [contributing](CONTRIBUTING.md), [architecture](docs/ARCHITECTURE.md),
[migration from prior models](docs/MIGRATION.md), [release checklist](RELEASE_CHECKLIST.md),
[publishing procedure](PUBLISHING.md) and [versioning](VERSIONING.md).
Bug reports and feature proposals have dedicated GitHub issue templates;
contributions follow the [code of conduct](CODE_OF_CONDUCT.md). The previous
project history and existing Git repository should be preserved when transferring
this staged source tree.

## Scope and limitations

MorphTree is an evolving-tree **research library**, not an established superior replacement for other tabular learners. The currently qualified native kernels accelerate selected operations but do not cover every step of the search; the learning-program controller remains in Python. Evidence of equivalent weights and decisions on tested inputs is **not** an independent demonstration that morphing improves generalization. Some native paths consume more memory. New large-scale, cross-platform and public-dataset benchmarks must be conducted before expanding performance or portability claims.

## Release status

- **Local technical qualification:** GO **only within the documented scope**.
- **GitHub transfer, hosted CI, TestPyPI and publication:** subject to separate owner-controlled approval and verification.
- **Package version:** `2.1.0`, approved for this candidate. This does not authorize uploading it.

## License

MorphTree is created by Luca Lullo and released under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
