# Comparison with MorphTree2.1.0 — final local evidence

These timings were recorded on the qualified integrated025 implementation,
before the final version/metadata and early buffer-pinning correction. The C
kernels and learning rules are unchanged; retained-model zero-fit checks cover
the correction, but no new fit timing is claimed for it. Any additional timing
campaign on the definitive commit requires its own approved budget.

All compared weights, parameters, seeds, tree nodes, variants, diagnostic state,
semantic lineage, decisions, probabilities and predictions are exact within the
same OS/interpreter. New Windows3.12 small/large pairs use contemporaneous actual
original2.1.0 references; no historical model was substituted.

| Input | Task | Python fit s | C fit s | Observed ratio | Python CPU s | C CPU s | Python peak MiB | C peak MiB | External score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|10003×15|binary|11.672|5.159|2.262|11.578|5.172|64.71|69.32|0.70094858|
|10003×15|multiclass|20.504|10.169|2.016|20.406|10.094|69.44|74.80|0.64553170|
|10003×15|regression|16.383|8.953|1.830|16.094|8.547|64.36|69.98|0.69977881|
|200003×15|binary|145.633|88.867|1.639|141.812|88.250|295.55|296.05|0.70494259|
|200003×15|multiclass|283.283|135.572|2.090|280.672|131.109|357.84|361.98|0.63904144|
|200003×15|regression|105.367|91.007|1.158|103.938|79.828|308.62|306.91|0.62136534|

External score is accuracy for classification and R² for regression, on2,003
independently generated EXTERNAL rows. It is unchanged by native reconstruction.
This is one ordered pair per task, not a randomized/repeated speed estimate.
No universal accuracy, speed, lower-RAM or HGB superiority claim follows.
Peak MiB is OS-maintained working-set peak of sampled owned worker/launcher PIDs,
not an absolute aggregate job RSS bound. Full fit includes preprocessing,
screening, OOF/introspection, descendants, Optuna and full-input refit.

Medium-scale Windows25,003×7 evidence spans Python3.12/3.13/3.14. Eight matched
recorder pairs observed1.58–1.80× public-fit acceleration; one earlier-recorder
binary pair is descriptive. Native worker peak was3–5MiB higher on that scope.
All102intermediate components matched exactly. Linux uses nine retained same-OS
references for exact correctness; their old times are not a contemporary speed study.

Retained-model full public inference passed25,003/200,003/1,000,003rows on
Windows3.12. At1,000,003periodic external rows, Python→C times were36.27→15.91s
binary,48.45→24.54s multiclass,40.11→19.63s regression, exact outputs and296MiB
OS-maintained worker peak. No new fit occurred; this is not million-row training.

Input, screening, diagnosis and mutation event durations are measured, but they
overlap component-fit intervals. Do not add them as disjoint costs. NUMERIC
component intervals include numeric learner work, not exclusive Optuna scheduler
time. Unseparated overhead remains inside the measured whole fit. Histogram
bridge measurements include buffer conversion; microkernel gains alone do not
establish whole-pipeline benefit.
