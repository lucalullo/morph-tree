# Integrated architecture

Public classes and original learning mathematics remain Python. The C ABI
implements ordered histograms and packed tree traversal. The binding validates
owned buffer types, lengths, indices, opcodes, graph structure, source and library
hashes. It never compiles at runtime.

Python preserves screening, one parent, TRAIN-only OOF diagnosis, executable
descendants, acceptance/rollback, continuation, frozen-program Optuna and final
full-input refit. Gradients/Hessians, split gain/tie ordering, leaf updates and
growth policies are unchanged. No new scientific rule or vendor engine is added.

ctypes.PyDLL retains the GIL. An RLock guards a4MiB packed-tree cache keyed by
complete bitwise node content, including signed zero. Batches below128rows use
equivalent Python traversal; larger supported calls use C. Native errors are not
hidden by silent fallback. MSVC /fp:strict or GCC -fno-fast-math -ffp-contract=off
preserve ordered binary64 arithmetic.

Exact Python/C comparisons apply within the same OS/interpreter. Original Python
also exhibits tiny CRT/glibc math differences across systems, recorded separately.
Models remain typed JSON. Standalone inference is standard-library Python;
training exports are distinct and require the exact runtime manifest.
Historical canonical/1.0 documents are retained as historical descriptions.

## Automatic numerical capacity and reported cost

The default screening and structural search start with two estimators, five
leaves and depth three. After the program is frozen, `n_trials='auto'` runs two
Optuna trials. Unless `capacity` explicitly supplies `n_estimators`, their
estimator domain depends on input shape and the configured time and memory
allowances. It is bounded by the existing 4,096-estimator parameter contract.
This is a maximum for a candidate, not a limit on input rows or a fixed default
model size. The 150-second minimum automatic allowance and the allocation of
45% of its initial value to numerical candidates and refit can make that upper
bound reachable on small inputs. The domain uses no target values or scores.
An automatic deadline extension covers the resulting work forecast; it does
not expand the domain again. These are workload heuristics, not guarantees of
execution time, generalization or optimal model size.

`report()['scalar_trees']` and `cost_report()['completed_scalar_trees']` describe
the cumulative training work across screening, OOF, descendants, numerical
trials and final refit. They are not the number of trees in the selected model.
For multiclass tasks, each boosting round contributes one scalar tree per class.
The selected model's numerical settings appear in `best_params_`,
`get_recipe()` and `available_variants()`. The phase events in `report()` give
the actual breakdown; `cost_report()` reads those events without new training.

An independent review reported 6,650 completed scalar trees for a binary fit of
180 rows and about 1.17 MB of saved JSON. Those totals alone do not demonstrate
a defect. For example, 12 structural/OOF trees, numerical trials of 2,999 and
640 trees, and a final 2,999-tree refit sum to 6,650 within the existing rules.
This is a possible accounting example, not a reconstruction of that particular
run: its feature shape, resource plan, seed, trial parameters and receipts were
not supplied with the observation. No new training was run for this review.

The ordinary `save()` bundle includes the selected model and any genuinely
retained `pure`/`tuned` variants, distinct weight packets only once, plus the
recorded history and diagnostics. JSON size therefore is neither peak RAM nor
the size of every model trained during search, and cannot be inferred solely
from the cumulative tree counter. A selected-only export is already available
through `save(path, variant='selected')` when only that state is needed.

Possible future work includes studying a more conservative numerical domain on
small data, validated stopping rules, and lossless storage reductions. Changing
capacity or stopping rules can change predictions, selection and generalization;
it requires a separately approved scientific comparison. This corrective delivery
changes no capacity default, Optuna rule, mutation or acceptance criterion.
