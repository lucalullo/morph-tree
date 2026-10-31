# Capacity comparison: 2, 4 and 8 rounds

CAPACITY132 completed 54 training runs with the installed MorphTree 2.1.0 native wheel on Windows x64 / Python 3.12. Each of six datasets used three paired seeds (42, 123 and 2026) and explicit capacities of 2, 4 and 8 rounds. This is a small comparative pilot, including three synthetic datasets; it does not establish a universally optimal capacity.

All runs retained structural evolution, two mutation proposals, two final Optuna trials, native acceleration and full-TRAIN refitting. Each run had the same 90-second fitting allowance and memory policy. Higher capacities consumed more trees and computation: equal time ceilings do not imply equal computational work. The actual number of scalar trees also depends on the task, selected family, descendants, OOF and final numerical choice.

Each seed used a 60% TRAIN / 20% VALIDATION / 20% TEST split. Internal SEARCH and OOF remained within TRAIN. Every model was frozen before VALIDATION; every validation result was frozen before TEST. Recommendations were calculated from preregistered VALIDATION conditions, not from TEST. No automatic default was changed.

## Held-out results

The following are TEST means over three paired seeds. Balanced accuracy is higher-is-better; log-loss and RMSE are lower-is-better. RMSE values from different datasets have different units and must not be averaged together.

| Dataset | Metric | 2 rounds | 4 rounds | 8 rounds |
|---|---|---:|---:|---:|
| Breast cancer, binary | Balanced accuracy | 0.944775 | 0.944775 | 0.948413 |
| Breast cancer, binary | Log-loss | 0.173173 | 0.143182 | 0.109538 |
| Synthetic binary | Balanced accuracy | 0.716889 | 0.767041 | 0.807485 |
| Synthetic binary | Log-loss | 0.479133 | 0.409423 | 0.358030 |
| Iris, multiclass | Balanced accuracy | 0.900000 | 0.922222 | 0.911111 |
| Iris, multiclass | Log-loss | 0.265232 | 0.170534 | 0.195954 |
| Synthetic multiclass | Balanced accuracy | 0.743978 | 0.802464 | 0.832981 |
| Synthetic multiclass | Log-loss | 0.571364 | 0.478542 | 0.402825 |
| Diabetes, regression | RMSE | 57.144186 | 55.806832 | 55.311284 |
| Synthetic regression | RMSE | 3.136374 | 2.639940 | 2.097683 |

Breast cancer has 569 rows and 30 features, Iris 150 rows and four features, and Diabetes 442 rows and ten features. Each synthetic dataset has 4,096 rows, 12 numeric features and one categorical feature. These are small datasets, so this experiment does not newly qualify large-scale performance. Three seeds reuse the same dataset with overlapping observations; they are not three independent populations.

## Costs

Values below are mean wall-clock seconds for the complete `fit()` call, followed by milliseconds for TEST inference. Classifier inference measures `predict()` and `predict_proba()` together; regressor inference measures `predict()` only. These short, single-run timings are descriptive, not a dedicated microbenchmark.

| Dataset | Fit seconds: 2 / 4 / 8 | Inference milliseconds: 2 / 4 / 8 |
|---|---:|---:|
| Breast cancer | 0.596 / 1.004 / 1.658 | 27.490 / 27.892 / 24.674 |
| Synthetic binary | 0.996 / 1.361 / 1.655 | 68.331 / 70.735 / 69.426 |
| Iris | 0.347 / 0.485 / 0.596 | 5.348 / 5.717 / 7.343 |
| Synthetic multiclass | 1.062 / 1.833 / 3.170 | 66.379 / 70.709 / 78.041 |
| Diabetes | 0.445 / 0.827 / 1.769 | 5.144 / 7.324 / 9.892 |
| Synthetic regression | 0.964 / 1.354 / 2.480 | 34.781 / 30.174 / 30.282 |

Across the 18 runs at each capacity, complete-fit wall totals were 13.230 / 20.591 / 33.983 seconds; process CPU totals were 11.594 / 17.766 / 30.266 seconds. The largest observed per-PID training working-set peaks were 155.145 / 155.449 / 155.570 MiB. These include interpreter and library memory and are not model-only memory or aggregate RSS limits. One Diabetes seed at capacity 8 took 6.50 times the complete-fit time of its capacity-2 counterpart; other paired timings did not exceed the preregistered 4-times warning. No paired working-set ratio exceeded 2.5.

Across the 54 TEST evaluation jobs, maximum observed per-PID working sets ranged from 149.980 to 153.504 MiB. This includes interpreter, loading, data, model, inference and metrics for the complete evaluation worker; it is not memory attributable solely to `predict()`.

Final whole-campaign conservative accounting was 1,468.25 wall seconds and 1,247.78125 CPU seconds. The executor ended at 1,408.25 wall seconds and 1,187.78125 CPU seconds, already including conservative preparation, process supervision and evaluation; closure then charged the reserved additional 60 wall seconds and 60 CPU seconds. This must not be confused with the sum of measured `fit()` calls. The completed budget was 54 public calls, 980 native fits including 244 OOF fits, 108 Optuna trials and 4,172 scalar trees, within the authorized ceilings.

## Recommendations and limitations

The preregistered classification condition required, on both datasets of each task, a mean paired VALIDATION balanced-accuracy gain of at least 0.005, no mean log-loss increase above 2%, and no seed balanced-accuracy deterioration above 0.01. The regression condition required at least 5% mean paired relative RMSE improvement on both datasets, with no seed deterioration above 5%. These conditions identify capacities worth further evaluation; they are not automatic promotion rules.

- **Binary classification:** both 4 and 8 passed the VALIDATION condition. Higher capacity helped the synthetic task strongly. On Breast cancer, the substantial validation improvement translated into unchanged mean TEST balanced accuracy at 4 and a small gain at 8; a seed deteriorated at each higher capacity. This supports further evaluation, not an unconditional default change.
- **Multiclass classification:** neither higher capacity passed on both datasets. Iris had unchanged mean validation balanced accuracy, a deteriorating seed, and capacity 8 increased mean validation log-loss by about 10.49%. The synthetic task improved substantially. There is no robust universal choice here.
- **Regression:** neither higher capacity passed on both datasets. Diabetes mean paired validation RMSE improved by about 1.55% at 4 and 4.37% at 8, below the preregistered 5% threshold. Synthetic regression improved by about 15.84% and 32.12%. On TEST, capacity 8 still worsened Diabetes RMSE for one seed.

An explicit `capacity={"n_estimators": 2}` is **not the complete current automatic policy**. The automatic resource-dependent final numerical domain can differ from an explicit cap. CAPACITY132 contains no full-automatic-policy control, so it does not directly establish that a fixed capacity of 4 or 8 would improve the ordinary no-capacity API. Keep the qualified defaults and offer explicit 4/8-round configurations when appropriate; evaluate a default change against the actual automatic policy before adopting it.

There were 108 genuine mutation decisions: 34 acceptances and 74 rollbacks. Native fitting, final refit and Optuna after program freeze were recorded. These findings verify execution and decision handling, but all arms evolve: without a static arm, accepted mutations do not demonstrate their causal contribution to held-out generalization. Likewise, two Optuna trials exercise startup sampling and do not establish an adaptive TPE advantage.

No adaptive capacity selector or new numerical, diagnostic or evolutionary policy was introduced. Dataset-level variation, short timings and the absence of an automatic-policy control limit any claim of a universally superior default.
