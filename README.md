# MorphTree

**Native tree learning with introspection and structural evolution.**

MorphTree is an experimental Python library for **binary classification, multiclass classification and regression** on tabular data. Instead of stopping at the selection of a fixed learner, it compares its qualified native tree-learning families, selects a single parent and examines where that parent makes errors. It then evaluates executable changes to the parent's learning rules, retains improvements that meet its validation criteria and rolls back changes that do not.

The evolutionary search runs within a finite budget. Once it stops, MorphTree freezes the selected learning program and, by default, uses **Optuna for final numerical tuning**. The result is a trained model ready for prediction, accompanied by its program, recipe, mutation history and search report.

The proprietary learning engine is written entirely in **Python**, without compiled extensions of its own. Third-party dependencies may contain compiled components. Structural evolution is implemented and tested, but independent predictive gains from mutation have not yet been established.

## Highlights

- One public `MorphTree` API for binary, multiclass and single-target regression.
- Native boosting and bootstrap bagging tested as applicable initial families; **one selected parent** proceeds.
- Error diagnosis, a typed **depthwise structural mutation**, child evaluation, acceptance/rollback and a finite evolutionary loop.
- Freeze before optional **numerical-only Optuna** finalization (`use_optuna=False` disables only tuning).
- Prediction, bounded JSON save/load, algorithm/recipe/history/report inspection and standalone **inference** export.
- Reproducible recorded decisions, conservative default budget and published limitations.
- No XGBoost, LightGBM, CatBoost or scikit-learn learners as internal training engines.

## Installation

Use **Windows x64 with Python 3.12** for the environment covered by retained qualification. From this repository's root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install .
.venv\Scripts\python -m pip check
.venv\Scripts\python -c "import morphtree; print(morphtree.__version__)"
```

Optuna `5.0.0` is a declared dependency. The command `pip install morphtree==1.0.0` is only appropriate **after** this exact distribution is released to an index. See [installation](docs/INSTALLATION.md) and [dependencies](DEPENDENCIES.md).

## Quick start

```python
from morphtree import MorphTree

model = MorphTree(random_state=42, verbosity=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification only:
probabilities = model.predict_proba(X_test)
```

If automatic task detection is ambiguous, supply `task="binary"`, `task="multiclass"` or `task="regression"` explicitly. The same evolutionary pipeline applies to each task; regression does not offer `predict_proba()`.

- [Binary classification example](examples/binary.py)
- [Multiclass classification example](examples/multiclass.py)
- [Regression example](examples/regression.py)
- [API reference](docs/API_REFERENCE_1_0.md)

## How MorphTree works

```text
Tabular data X, y
       |
       v
Test qualified applicable native families
       |
       v
Select ONE parent -> diagnose errors
       |
       v
Mutate executable learning rules -> train and evaluate child
       |
       +-- justified improvement --> accept as new parent --+
       |                                                     |
       +-- otherwise -------------> rollback                 |
       |                                                     |
       +<---------- repeat within finite budget <------------+
       |
       v
Freeze learning program -> optional numerical Optuna -> model + history
```

A mutation changes the executed growth rule, not merely a parameter. The present operator set is **deliberately limited**; descendants require retraining. A valid run may accept **no** mutation and retain its parent. [Canonical definition](docs/Relazione_MorphTree.txt) · [Architecture](docs/ARCHITECTURE_1_0.md).

## Parameters and model inspection

Defaults include a **150-second cooperative time budget**, one worker, two initial stages/members, up to two structural attempts and two final numeric trials. Controls include `task`, `random_state`, `verbosity`, `time_budget`, `max_mutations`, `n_trials` and `use_optuna`. See the API reference for the exact accepted arguments and bounds.

```python
model = MorphTree(task="regression", random_state=42, use_optuna=False)
model.fit(X_train, y_train)
model.save("model.json")
restored = MorphTree.load("model.json")
print(restored.describe_algorithm())
program = restored.get_algorithm_program()
recipe = restored.get_recipe()
history = restored.get_mutation_history()
report = restored.get_search_report()
restored.export_standalone("predictor.py")
```

`export_standalone()` exports inference only. `export_algorithm()` exports a bounded training-program fragment requiring a matching MorphTree runtime, **not** the autonomous search as a standalone trainer.

## Evidence and limitations

Authorized earlier LIVE campaigns exercised real training, out-of-fold components, mutation proposals, **correct rollback**, freeze, Optuna and model I/O for all three tasks on small synthetic inputs. Observed descendants in those campaigns did **not** qualify as improvements. Accepted-child continuation has software-test evidence, **not** independent LIVE proof of better generalization.

The current product scope is conservative: dense tabular inputs, single-label classification / single-target regression, bounded feature and payload sizes, a limited native portfolio and a single qualified depthwise mutation operator. Training cost includes **every** initial family trial, diagnosis, OOF component, mutation and Optuna trial. Timeout/memory controls are partly cooperative and **do not guarantee** an absolute peak RSS cap. Neither predictive competitiveness nor independent adaptive benefit has been established.

See [benchmarks and disclosure](BENCHMARKS.md), [research](research/README.md), [release qualification](docs/RELEASE_QUALIFICATION.md) and [roadmap](ROADMAP.md).

## Documentation

- [Installation](docs/INSTALLATION.md) · [Public API](docs/API_REFERENCE_1_0.md)
- [Architecture](docs/ARCHITECTURE_1_0.md) · [Canonical project specification](docs/Relazione_MorphTree.txt)
- [Research summary and experiment registry](research/README.md)
- [Release qualification](docs/RELEASE_QUALIFICATION.md) · [Manual CI instructions](docs/CI_PREPARATION.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Changelog](CHANGELOG.md)

The public research section is **curated**. The extensive original experiment reports, internal tools and private evidence remain in the separate full development archive for Codex. Research records include negative and inconclusive outcomes; their IDs are retained without claiming new independent studies or earlier official releases.

## License

Released under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).