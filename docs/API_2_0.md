# MorphTree 2.0.0 task estimators

This is the API of the local 2.0.0 development target. Source implementation,
unit tests, installed-artifact tests and LIVE qualification are separate gates.
The approved bounded Windows and Ubuntu24.04WSL2/Python3.12.3 profiles have passed three-task LIVE checks;
Python3.15 is experimental; official2.0 support is Windows3.12–3.14 and
Ubuntu24.04 WSL2/Python3.12. See
[the exact qualification and blockers](RELEASE_BLOCKERS_2_0.md).

```python
from morphtree import MorphTreeClassifier, MorphTreeRegressor

classifier = MorphTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
labels = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

regressor = MorphTreeRegressor(random_state=42)
regressor.fit(X_train, y_numeric)
values = regressor.predict(X_test)
```

Both estimators use the existing MorphTree engine, native learners, admission,
finite budgets, diagnostic acquisition, structural evolution and final numeric
driver. The classifier selects only binary versus multiclass, using two versus
3–64 distinct labels. Numeric classification labels remain labels. Regression
never infers classification from integers or from two distinct numeric values.
The public task estimators have no `task` constructor parameter.

## Controls and state

Shared keyword controls are `random_state=None`, `verbosity=0`,
`use_optuna=True`, `max_mutations=2`, `time_budget="auto"`, `n_trials="auto"`,
`n_jobs=1`, `capacity=None`, `mode="adaptive"`, `operators=None`.
The existing engine validates them. Default evolution remains enabled when
`use_optuna=False`; that flag disables only final numeric optimization.
`mode="baseline"` explicitly opts out of structural evolution. Explicit research
operators still require their existing research context and are not ordinary
default-user capabilities.

`get_params()` returns canonical constructor controls without task or internal
learner configuration. `set_params()` validates a replacement transactionally;
invalid values preserve the prior estimator, while a valid update clears fitted
state. `fit()` returns the facade itself. Fitted `task_`, `n_features_in_`,
optional `feature_names_in_`, `best_family_`, `best_params_`, `stop_reason_` and
classifier `classes_` reflect the engine. Mutable attribute views are defensive
copies. `score(X,y)` returns accuracy for classification and R² for regression,
using the existing finite constant-target convention for R². Regressors do not
offer `predict_proba`. scikit-learn Pipeline/CV compatibility is not claimed.

## Automatic fit and the observed outcome

`fit(X, y)` prepares supported data, screens both qualified native families,
selects one parent, obtains current-parent OOF diagnostics and tries admissible
structural descendants. It retains verified improvements or rolls back, then
freezes the program before final numeric Optuna. No manual family, mutation or
preprocessing selection is required. Defaults are150 seconds, one worker, two
mutation opportunities and two numeric trials; these are caps, not promises
that a mutation will be accepted or every phase will complete.

After fit, inspect the actual result rather than infer it from the constructor:

```python
outcome = classifier.report()
print(outcome["tuning"], len(outcome["trials"]))
print(classifier.stop_reason_)
print(classifier.get_mutation_history())
```

`OPTUNA_CALLBACKS_COMPLETE` means the numeric callbacks completed;
`SKIPPED_SEARCH_BUDGET` and `OPTUNA_STOPPED_SEARCH_BUDGET` describe incomplete
numeric work. Disabling Optuna explicitly yields `DISABLED`. Errors are not
converted into a claim of successful optimization. `get_search_report()` is a
separate verified cost/lineage projection; a loaded report is marked persisted
and is not presented as newly verified journal evidence.

## Inputs, missing values and limits

Inputs are rectangular in-memory lists/tuples, dense NumPy arrays or pandas
DataFrames. Features support finite numeric values, booleans, strings and missing
NaN, None or pandas.NA, including pandas nullable numeric/string/boolean dtypes.
Sparse arrays, arbitrary nested objects and datetime preprocessing are not part
of this input contract. pandas is optional. NumPy is also an Optuna transitive
dependency; accepting NumPy arrays does not add a separate native ML engine.

The internal TRAIN/SEARCH split fits categorical vocabulary on TRAIN only.
Each categorical column has separate known, missing and unseen channels.
Maximum categorical vocabulary is 64 entries per column; encoded public width
is at most 128. The current public path estimates at most 2 MiB of input payload
and permits at most 65,536 TRAIN rows. A column entirely missing in TRAIN is
numeric; subsequent string values in that column raise an explicit error.
Known numeric columns do not silently become categorical during prediction.
Named input must retain the same unique column identities; column order can be
realigned. Positional and named schemas cannot be silently interchanged.

Targets must be homogeneous finite numbers or strings for classification and
finite numbers for regression. Missing targets, mixed label types and multioutput
targets are rejected. At least four observations are needed; classification
requires at least two per class for disjoint TRAIN/SEARCH. Duplicate observation
groups are rejected until a group-preserving split policy is qualified.
Controlled/sealed research inputs retain the existing guarded context boundary.

## Persistence and inspection

```python
from hashlib import sha256
from pathlib import Path

# classifier is the already fitted object from the example above.
model_path = Path("classifier.json")
classifier.save(model_path)
expected_sha256 = sha256(model_path.read_bytes()).hexdigest()
restored = MorphTreeClassifier.load(model_path, sha256=expected_sha256)
assert restored.predict(X_test) == classifier.predict(X_test)
assert restored.predict_proba(X_test) == classifier.predict_proba(X_test)
restored.export_standalone("predictor.py")
program = restored.get_algorithm_program()
history = restored.get_mutation_history()
recipe = restored.get_recipe()
report = restored.get_search_report()
```

Keep the JSON file and its expected digest for later use or transfer. For a
received model, obtain the expected digest from a trusted source; hashing an
untrusted file by itself does not establish authenticity. The same save/load
pattern applies to `MorphTreeRegressor`, without `predict_proba`. Prediction
and loading do not retrain or contact a service. The model JSON contains learned
weights, preprocessing/schema and the selected recipe; it uses an installed
compatible MorphTree runtime. Standalone export supplies inference code and
its learned state, not an independently retrainable search procedure.

JSON validation, size limits, optional caller-provided SHA-256, native model
payloads and standalone preprocessing use the existing persistence machinery.
No new model envelope or executable pickle is introduced. Task-specific loading
rejects an incompatible regression/classification model. A loaded classifier
retains classification intent for a future separately authorized refit.
`report`, `cost_report`, `available_variants`, `save_recipe`, `export_python` and
`export_algorithm` delegate to the same engine. Standalone export is inference
only; training-program exports retain their separate execution requirements.
Persisted reports do not become newly verified journal evidence on load.

Legacy `MorphTree` and its old safe JSON model envelopes remain available.
The old root-level low-level classifier/regressor classes are now exposed as
`NativeMorphTreeClassifier` / `NativeMorphTreeRegressor` and remain in
`morphtree.model`; their existing training admission still applies. Code using
their `genome/config/params/backend` constructor must migrate to those names.
The new public estimators expose the full evolutionary program instead.

## Qualification and release boundary

Official2.0 support covers Windows AMD64/CPython3.12–3.14 and Ubuntu24.04
WSL2 x86_64/CPython3.12. Python3.15 and missing platforms remain subsequent
qualification objectives. Each declared interpreter/system needs
its own install, dependencies, preprocessing, LIVE pipeline, persistence and
integrity evidence. An import or a synthetic test does not establish that gate.
Independent predictive gain, universal beneficial mutations, full causal
diagnoses and absolute process RSS bounds are not claimed. A correctly rejected
mutation is a valid evolution outcome. Broader scientific ambitions continue
after the first reliable release, with the native mutating engine preserved.
