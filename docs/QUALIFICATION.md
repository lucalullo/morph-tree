# Qualification — MorphTree 2.1.0
2.1.0 is the owner-selected native package version. Local qualification does
not establish package-index availability or publication approval.
Learning implementation and C sources are identical to the frozen native092
engine except package/API version literals and reviewed metadata. Separate RC
installed checks and any new training are recorded in RELEASE_STATUS.json.

Historical integrated lineage:Windowsx64 CPython3.12 three tasks at10,003×15,
25,003×7,200,003×15;Windows3.13/3.14 and Ubuntu24.04WSL2x64/glibc2.39
3.12–3.14 three tasks at25,003×7. These are retained results,not new RC fits.
092 installed wheel/sdist and historical-model checks cover Windows/Linux
3.12–3.14. SCIENCE099 ran45controlled pipelines across three synthetic
distributions/three seeds; it did not establish general evolutionary superiority.
REAL108 and PREDICTIVE110 are frozen stellar results. Capacity8 improved only
the reported VALIDATION; four proposed structural changes all rolled back.
OBSERVATION112 acquired two genuine OOF models/48trees with exact old pointloss;
STARinM had low recall but no stable aggregate STARgradient bias in those folds.
PREDICTIVE089 had incomplete preflight; subsequent checks do not repair it.

Historical APIs,save/load,standalone,exports,refit and automatic backend have
scoped LIVE evidence. Reuse depends on source identity; metadata-only version
updates do not by themselves qualify a new physical fit. Fresh RC zero-fit
checks must verify actual installed artifacts and historical models separately.
Cross-OS tiny differences from CRT/glibc math.exp are known; exact comparisons
use the same OS/interpreter. Million-row inference is a retained Windows3.12
check,not million-row training. Compatible wheels need no compiler; sdist does.

Unqualified:Python3.15,authentic macOS,other Linux/manylinux,Linux large training,
million-row training,hosted Kaggle,arbitrary concurrent model mutation,universal
accuracy or speed superiority. Local checks do not replace future hosted CI on
the final transferred commit. Publication approval remains false.

## Fresh RC116/RC117 completion
Three default installedRCfits perOS,Windows3.12.10/UbuntuWSL23.12.3 completed.
PerOS3calls,34fit including10OOF,6Optunatrials,291scalar trees. Weights,split
programs,diagnoses,lineage,accept/rollback,probabilities and predictions are
exact against retained same-OSPython2.1references. Every25003input row is
accounted for in final refit;Optuna begins after freeze. Save/load,variants,
score,recipe,training export and standalone inference passed. No reference
retraining or tolerance relaxation. See RC_TRAINING_SUMMARY.json.

Final source differences after these fits are qualification labels,documentation
and corresponding test assertions only. Learning code and native binaries are
unchanged;final artifact checks record that transfer explicitly. FreshRCtraining
was NOT repeated on3.13/3.14:historical unchanged-engine evidence and freshRC
installation/inference tests are identified separately.

## Definitive2.1.0 version alignment
RC116/RC117 were physically trained as3.0.0rc1. Their evidence is historical,
not relabelled as new2.1.0 fits. This successor changes only two package/API
version literals,identity metadata,tests and documentation. Learning code and
C binaries remain unchanged. Fresh2.1.0 installs and any separately authorized
training are recorded explicitly. Historical Python-only2.1.0 remains distinct.

## Linux binary distribution audit
Auditwheel6.8.2 found onlyGLIBC_2.2.5 versioned symbols fromlibc and no extra
shared libraries to bundle. The repaired wheel carries manylinux platform tags,
including manylinux2014/manylinux_2_17. File-by-file validation confirms identical
MorphTree code and native ELF bytes:only WHEEL/RECORD container metadata changes.
This is a binary ABI/symbol/ISA audit,not a new complete-pipeline qualification
on older glibc or other Linux distributions. Actual runtime tests remain the
documented Ubuntu24.04WSL2x86_64/Python3.12–3.14 profiles. No compiler is needed
when installing the compatible precompiled wheel.

Linux builders can install the development-only tools auditwheel==6.8.2,
pyelftools==0.33,packaging==26.3,patchelf==0.19.1.0 into an isolated environment,
along with build==1.3.0/hatchling==1.27.0,then run:

    python -I -B tools/build_artifacts.py --manylinux

The build retains its raw wheel and audit logs in its owned temporary copy.
It refuses repair if any code/library/other file payload changes;no new runtime
dependency is added. Standard source builds still require a local C toolchain.

## Fresh2.1.0 training completion
VERSION121/122 each completed3installed facade calls,34native fits including
10OOF,6trials and291scalar trees. All same-OS reference weights,predictions,
probabilities,diagnostics and evolutionary decisions are exact. Binary accepts
two successive generations,multiclass rolls back,regression accepts then rolls
back. Optuna follows freeze;all25003input rows enter final refit. Save/load,
variants,score,recipe,training exports and standalone inference pass.
Final label-only successor changes qualification JSON/test assertions/documents,
not learning code or compiled libraries;final installed checks bind that transfer.
No new pipeline fits on3.13/3.14 are implied by their clean installation checks.

## Public task API versus legacy admission

The public package exports the classes in `morphtree.scalable`; its final imports
supersede the older task wrappers. The 2 MiB/65,536 TRAIN-row/128 encoded-feature
checks in `api._new_user_session` belong to the separate legacy API. They do not
limit the current public task classes. The large supervised comparisons called
those public classes; their wrappers counted native fits, saved receipts and
checked stop conditions while invoking the original learner. They did not remove
input caps or replace preprocessing, selection or evolution.

## NumPy prediction boundary correction

Public task prediction outputs now use NumPy arrays, including historical-model
facades. Learning, splitting, diagnosis, acceptance, numerical tuning and C
sources are unchanged. Qualification of this correction uses installed zero-fit
comparisons against retained predictions, exact model packets and genealogy,
both Python and native inference, serialization and explicit output formats.
The previous LIVE runs remain historical training evidence; this correction does
not claim new physical fits. Default dependency-free exports and CLI JSON retain
their list/JSON contracts; explicit NumPy exports use output_format='numpy'.

Completed supervised large training, zero-fit admission, metadata-only shape
tests and ordinary unattended completion are different claims. No additional
physical training was run for the inventory/documentation correction. See
[the detailed resource guide](PUBLIC_API_LIMITS.md); retained LIVE evidence stays
historical and conditional on unchanged runtime source identity.


## Current public description
The README revision authorized by the owner describes the native implementation and its measured
limitations without declaring hosted CI or Kaggle qualification. README_REVIEW.md
records the current source hash, metadata checks and remaining release gates.
Historical RC labels and qualification records above identify their actual runs;
they are not relabelled as newly executed final-package training.

## CAPACITY132: completed physical training

The installed RELEASE131 native wheel completed 54 complete fits: six datasets,
three seeds and explicit capacities 2/4/8, on Windows x64 / CPython 3.12.10.
All three tasks, native histogram use, final refit, OOF, Optuna after program
freeze and 108 structural decisions (34 acceptances, 74 rollbacks) were recorded.
Consumption was 980 native fits including 244 OOF, 108 trials and 4,172 scalar
trees. All 54 models and validation assessments were frozen before TEST.
See [the comparative results and limitations](CAPACITY.md). No automatic default
changed and no causal evolutionary gain is inferred without a static control.

The subsequent README/documentation revision changes no runtime Python or C
source, default or numerical rule. Its rebuilt artifacts require fresh packaging
and installed software checks; it does not relabel CAPACITY132 as new Linux
training or as a fresh fit of a differently named artifact. Runtime equivalence
and artifact hashes are recorded separately in the local delivery report.
