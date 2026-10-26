# Executable evolving learner

## Permanent Pure-Python implementation contract

MorphTree's proprietary implementation is Python throughout: tree induction,
split search, objective derivatives, preprocessing, diagnostics, executable
structural mutations, descendant evaluation, acceptance/rollback, continuation,
final numeric orchestration and inference. C/C++, Rust, Cython, custom binary
extensions and JIT compilation are prohibited, including for speedups. Improve
algorithms, data structures, repeated computation and memory in Python.

The public evolutionary path explicitly uses `PythonBackend`, whose arithmetic
uses the standard library. The legacy explicitly selected `numpy_backend.py`
is Python source calling external NumPy array operations; it is not the default
public evolutionary engine and is not described as standard-library-only.
NumPy input support and this legacy API remain preserved.

External packages must be distinguished from proprietary code. Optuna requires
NumPy and other distributions with upstream binary wheels. Python fallbacks
exist for PyYAML and sampled SQLAlchemy/MarkupSafe primitives, but partial
fallback checks do not qualify wholesale dependency replacements. Normal pip
installation must work without a user compiler or manual build flags. The
ordinary Windows3.15/PyYAML installation remains unresolved. The owner excludes
3.15 from official2.0 support; its laboratory evidence and future roadmap remain.

`ctypes` calls in the process guards access literal OS system libraries for
owned-process safety and accounting. They implement no compiled ML kernel and
must not be removed to satisfy a language-label check. Historical private C/JIT
research is preserved as inactive evidence; no binary/compiler activation is
available through the current laboratory routes or public distribution.

The audit leaves all49 runtime files unchanged. The existing bounded LIVE
evidence remains tied to those bytes; further runtime changes require their
own qualification and new approval before any additional training.

The initial native boosting/bagging comparison selects one parent. Only that
parent enters current-parent TRAIN-OOF diagnosis and typed structural evolution.
The executable AlgorithmProgram controls native behavior;recipe/numerical
parameters/model weights are separate. A depthwise proposal executes a different
growth rule,then SEARCH verifies it. Justified changes are accepted;ties/worse
candidates roll back. The loop stops finitely,freezes and tunes numbers with
Optuna by default (`use_optuna=True` in both public task estimators). Explicit
`use_optuna=False` disables only numeric tuning. The ordinary defaults are150
seconds, one worker, two mutation opportunities and two numeric trials. Budget
exhaustion is reported distinctly from completed tuning. The learner can stop
without an accepted mutation; future scientific operators are not release gates.

Current children require full retraining. Warm start and a second distinct
operator are not implemented. Positive accepted-child continuation has software
evidence;the retained LIVE runs did not demonstrate beneficial acceptance.
OOF are fold-disjoint and encoder-local,but do not make the entire adaptively
selected procedure an unbiased validation estimate. GUARD never directs search.

The bundled private namespace `_experimental` is retained for format compatibility;
it is self-contained package code,not a lookup into a research checkout.
Distribution1.0.0/native format0.6 remain distinct. No XGBoost,LightGBM,CatBoost
or sklearn learner is an internal engine. Read the exact canonical relation in
docs/Relazione_MorphTree.txt and the stable vision in docs/MORPHTREE_VISION.md.
