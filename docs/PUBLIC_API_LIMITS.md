# Public task API and resource limits

`from morphtree import MorphTreeClassifier, MorphTreeRegressor` resolves to
`morphtree.scalable`. The final imports in `python/morphtree/__init__.py` replace
the earlier task-wrapper names. The scalable classes call their own `fit` and
`_fit`; they do not call `api._new_user_session`.

| Path | Input admission |
| --- | --- |
| Public task classes in `morphtree.scalable` | Shape/workspace forecasts, configured time/memory allowance and observed resource headroom; no 2 MiB,65,536 TRAIN-row or128 encoded-feature cap |
| Legacy `morphtree.api.MorphTree` and older task wrappers | Retain those fixed admission caps for their separately constrained execution path |
| Historical models loaded by the public classes | Retain the historical decoder/inference constraints until a new scalable fit succeeds; loading does not retrain or rewrite weights |

## What happens during ordinary fit

`Input` inspects dense shape/schema metadata first. `fit_plan` forecasts memory
and work; `Resources.reserve/check` rejects or interrupts work that exceeds the
allowance or observed headroom. Columnar input storage and fold-local categorical
vocabularies avoid whole-input NumPy/pandas conversion to nested lists. A category
column remains one nominal feature rather than one-hot expansion. Missing values
and unknown categories use the qualified encoder. Vocabulary storage, result
allocation, representable index types and the learner's numerical parameter
contract remain real constraints; there is no promise of unlimited cardinality,
columns, rows, RAM or execution time.

All eligible input rows enter final refit when `refit_full=True` (the default).
Internal TRAIN/SEARCH splitting, OOF folds and bagging are algorithmic operations,
not silent input downsampling. Prediction uses internal resource-aware blocks,
preserving order; returned predictions still need memory. Save/load also perform
resource and integrity checks. Sparse input and unsupported scalar/schema types
are rejected explicitly. See [API](API.md) and [architecture](ARCHITECTURE.md).

## What has and has not been demonstrated

- Retained supervised three-task Windows/Python3.12 fits cover200,003×15,
  including missing values and1,024 source categories. They invoked the public
  scalable class, with resource observers and receipt wrappers around the
  original native learner. These wrappers did not bypass admission or replace
  the algorithm. Source identity is required to reuse those results.
- VERSION121/122 cover the current2.1.0 runtime on25,003×7 under Windows and
  Ubuntu24.04WSL2/Python3.12. Later packaging-only changes do not become new fits.
- Deterministic tests can stop before column allocation or the first learner
  body to verify that a large shape passes admission. Such tests are **zero fit**;
  they do not demonstrate successful training, prediction accuracy or peak RAM
  for the unexecuted portion. Controlled resource fixtures are labelled as such.
- Ordinary `fit(X,y)` has the same scalable implementation, but no new unattended
  large training was run for this documentation correction. In particular, an
  automatic budget is a forecast, not a guarantee of completion on other data or
  hardware. Linux large training and million-row training remain unqualified.

The explicit eight-round configuration remains
`MorphTreeClassifier(random_state=42, capacity={"n_estimators":8})`, and the same
option works for regression. It changes neither the qualified default nor any
diagnostic, SEARCH, acceptance or Optuna rule.
