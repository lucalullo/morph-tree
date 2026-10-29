# MorphTree

**An evolving tree learner for tabular data, with a simple Python API and compiled C acceleration.**

MorphTree screens its own supported tree-learning families, diagnoses training errors, explores executable **structural changes to its learning program**, and retains descendants only when the evaluation criteria are satisfied. After structural search, it freezes the selected program, runs final numerical tuning with Optuna by default, and refits the selected model on the eligible input data.

**Version: 2.1.0 · Status: locally qualified candidate.** This repository is staged for the project owner's review and manual transfer. **It has not been approved for public publication.** The native and historical Python-only implementations are separately traceable by their source and artifact hashes.

## Quick start

```python
from morphtree import MorphTreeClassifier

model = MorphTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
model.save("morphtree-model.json")

restored = MorphTreeClassifier.load("morphtree-model.json")
```

For regression, replace `MorphTreeClassifier` with `MorphTreeRegressor` and use `predict()`; for multiclass classification, use the same classifier. `fit(X, y)` accepts supported tabular input without a separate task flag.

The standard workflow includes automatic handling of supported missing and categorical values. Internally, MorphTree plans resource use and handles inference batches; it does **not** silently drop rows or disable evolutionary search to fit an arbitrary row cap. Work can still be rejected when it cannot be admitted safely within available resources.

## What makes MorphTree different?

1. **Native-family screening.** Evaluate the applicable built-in tree learners and choose a starting parent.
2. **Error introspection.** Use out-of-fold TRAIN diagnostics to identify areas in which the parent struggles.
3. **Structural evolution.** Propose changes to the actual learning program, train descendants, verify the results, and accept or roll back changes.
4. **Multi-generation continuation.** Subsequent proposals build on the retained parent; acceptance is never forced.
5. **Final numerical tuning.** Freeze the program before Optuna tuning (`use_optuna=True` by default), then refit the selected model.
6. **Inspectable results.** Save and load models, inspect the learned program and genealogy, export recipes and standalone inference, and access recorded costs.

The present structural mutation grammar is deliberately finite. MorphTree is **not** claimed to outperform other learners in general, nor has the predictive benefit of introspection-guided evolution been independently established across diverse real-world benchmarks. See the [research record](https://github.com/lucalullo/morph-tree/blob/main/research/README.md) and [qualification scope](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md).

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

For classification, `score()` is accuracy. Optional controls include `max_mutations`, `use_optuna`, `n_trials`, `time_budget`, `memory_budget`, `mode`, `refit_full` and `n_jobs`. Supported advanced model inspection, checkpoint and export options are described in the [API guide](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md).

The normal adaptive mode keeps evolutionary search active. Explicit `mode="baseline"` is available for controlled comparisons and does **not** perform structural evolution. This distinction matters when evaluating results.

## Installation

The **2.1.0 native candidate is not yet published**. Install only a reviewed, platform-compatible wheel provided with the local delivery:

```bash
python -m venv .venv
# Activate the environment using your platform's standard command.
python -m pip install path/to/morphtree-2.1.0-py3-none-PLATFORM.whl
python -m pip check
python -m morphtree --help
```

Use the virtual environment's `python` after activation. The wheel filename above is a placeholder: select the actual Windows or Linux artifact, not the literal `PLATFORM` string.

- **Python:** CPython 3.12–3.14 on qualified 64-bit x86 systems.
- **Windows:** wheel built with MSVC, no compiler required for a compatible wheel.
- **Linux:** verified on Ubuntu 24.04 under WSL2; the available `linux_x86_64` wheel is **not** a universal manylinux portability guarantee.
- **Source installation:** requires a compatible C toolchain, a linker and system development headers; building from source compiles the native kernels.
- **Core dependency:** Optuna 5.0.0. Pandas and scikit-learn are optional integrations, not required for the basic NumPy-style workflow.

See the full [installation instructions](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md), [dependency record](https://github.com/lucalullo/morph-tree/blob/main/DEPENDENCIES.md) and [security notes](https://github.com/lucalullo/morph-tree/blob/main/SECURITY.md).

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

These are **individual measurements, not replicated statistical estimates**. They do not establish universal accuracy, speed or memory superiority; some native paths use more memory. Inference has been checked up to 1,000,003 rows, but **million-row training has not been qualified**. Large Linux training, Python 3.15, macOS, broad Linux portability and hosted Kaggle are also unqualified.

See [measurements](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md), [qualification details](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md) and [benchmark methodology](https://github.com/lucalullo/morph-tree/blob/main/BENCHMARKS.md) for caveats and evidence provenance.

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

See [contributing](https://github.com/lucalullo/morph-tree/blob/main/CONTRIBUTING.md), [architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md), [migration from prior models](https://github.com/lucalullo/morph-tree/blob/main/docs/MIGRATION.md) and [manual repository handoff](https://github.com/lucalullo/morph-tree/blob/main/docs/LOCAL_HANDOFF.md). The previous project history and existing Git repository should be preserved when transferring this staged source tree.

## Release status

- **Local technical qualification:** GO **only within the documented scope**.
- **GitHub transfer, hosted CI, TestPyPI and publication:** subject to separate owner-controlled approval and verification.
- **Package version:** `2.1.0`, approved for this candidate. This does not authorize uploading it.

## License

MorphTree is distributed under the [MIT License](https://github.com/lucalullo/morph-tree/blob/main/LICENSE).
