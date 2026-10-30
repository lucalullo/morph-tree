# Public API and compatibility

`MorphTreeClassifier` handles binary and multiclass classification;
`MorphTreeRegressor` handles numeric regression. Both expose `fit`, `predict`,
`score`, `get_params`, `set_params`, `save`, `load`, `report`, `get_search_report`,
`get_algorithm_program`, `get_recipe`, `get_diagnostics`, `get_mutation_history`,
`cost_report`, `available_variants`, `save_recipe`, `export_python`,
`export_standalone` and `export_algorithm`. Classification also has `predict_proba`.
`score` returns accuracy or R². `load_model` chooses the correct public task class.

The package-level classes resolve to `morphtree.scalable`, not the legacy
`morphtree.estimators` wrappers. The legacy input caps do not apply to the
scalable task classes. See [public API and resource limits](PUBLIC_API_LIMITS.md)
for resource admission, historical-model limits and the scope of large-data tests.

Defaults are `random_state=None`, `verbosity=0`, `use_optuna=True`,
`max_mutations=2`, `n_trials='auto'`, `n_jobs=1`, `mode='adaptive'`,
`operators=None`, `capacity=None`, `time_budget='auto'`, `memory_budget='auto'`,
`refit_full=True`, `checkpoint_dir='auto'`. Explicit integer seeds are repeatable;
None draws a fresh seed for each fit and records it as `effective_seed` in the
report. The constructor parameter remains None, including after save/load.
`n_jobs='auto'` selects the supported serial scheduler. It never spawns extra
workers outside the resource plan. `n_trials='auto'` resolves to two trials.

The mutation budget is a maximum attempt count. The current finite grammar uses
`split_program:objective_balance` and `growth_policy:depthwise`; a lineage has at
most three distinct proposals. Acceptance is never forced. Explicit `baseline`
uses boosting without structural evolution. Explicit `research` requires
`operators=('growth_policy:depthwise',)`. Adaptive retains both native families,
diagnosis, structural comparisons, rollback, continuation and final Optuna.

`get_params(deep=False)` preserves constructor object identity. Deep access is
defensive. Empty `set_params()` preserves fitted state; invalid updates are
transactional; valid changes reset fitted state. A new fit after load is a new
complete training. A failed refit preserves the prior usable model and records
the failed attempt separately. Saved artifact destinations are not restored as
automatic output paths.

`load(path, sha256=None, workdir=None, memory_budget='auto')` validates typed JSON,
model graphs and recorded integrity. `workdir` accepts an existing local directory
path or PathLike with no link/reparse ancestors. The historical decoder uses it
for its cache; the current in-memory decoder needs no additional disk cache.
Historical 2.0.0 models are real subclasses of the current public task classes,
with original weights, state, reports and inference. They retain their historical
engine limits until a new scalable fit succeeds. No old weights are fabricated
or silently converted into a new model.

`stop_reason_` describes the retained model: `PLANNED_SEARCH_COMPLETE` for normal
completed fits, `FROZEN_PROGRAM_TRAINING_COMPLETE` for frozen trainers,
`PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT` for recovered TRAIN anchors,
or `LEGACY_STOP_REASON_NOT_RECORDED` when no old reason was recorded. Historical
recorded reasons are preserved. A failed attempt does not replace the previous
model's reason. These are operational reasons, not proofs of optimality.

`save(path)` preserves genuine available variants. `selected` is the active
model, normally refitted on all eligible input. `pure` is the evolved TRAIN
model before numerical search. `tuned` exists only for a retained numerical TRAIN
winner. Neither alternate implies a second full-input refit. Explicit
`variant='selected'/'pure'/'tuned'` selects that actual state for save, recipes,
program inspection and inference export. Missing weights raise before writing
and never trigger training. `load_checkpoint(directory, stage='current'/'selected')`
recovers validated checkpoints; it does not resume training automatically.

`export_standalone(path)` exports a fitted standard-library predictor.
`export_algorithm(path)` exports a genuine frozen native family training fragment,
with resolved seed, parameters, encoder and classes, bound to the exact runtime
manifest. Its `train_encoded` method accepts compact encoded TRAIN data, internal
numeric targets and exact provenance. It performs a new native training operation:

```python
identity = {
    "data": data_identity(encoded.source, targets, "TRAIN"),
    "preprocessing_sha256": encoder_digest(encoder),
}
trained = exported.train_encoded(encoded, targets, identity=identity)
```

The advanced helpers are `Encoder`, `Table`, `Input` in `morphtree.scalable_data`,
`detect_target` in `morphtree.scalable_target`, and `data_identity`/`encoder_digest`
in `morphtree.scalable_receipts`. Use `Encoder.from_state(SPEC['encoder'], resources)`
and transform the TRAIN table; classification targets use native I/Q buffers,
regression uses d. Signed q classification buffers remain accepted after range
validation. The fragment provides finite resource checks and an optional `check`
callback. It does not repeat screening, OOF, evolution or Optuna. The historical
2.0.0 exporter retains its original admitted-Session fragment contract.

Optional `as_sklearn()` supplies NumPy predictions for supported sklearn
Pipeline/scorers. No universal `check_estimator`, sparse input, sample weights,
metadata routing or cross-validation qualification is claimed.

The console commands `morphtree inspect`, `predict`, and `export` accept
`--model` and optional `--sha256`; prediction takes `--input`, export takes
`--output`. `python -m morphtree` offers the same interface. Input JSON can be
positional rows or `{"columns": [...], "rows": [...]}`. The CLI uses the unified
loader and resource admission, including input beyond the historical 2 MiB cap.
