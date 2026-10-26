# MorphTree contributor contract

The proprietary engine must remain entirely Python: trees, splits, objectives,
preprocessing, introspection, structural mutations, evolution and inference.
Do not introduce C/C++, Rust, Cython, native extensions or JIT/Numba acceleration.
Optimize Python algorithms, data structures, repeated work and memory while
preserving precision, safety and every supported capability.

The default public evolving trainer uses the standard-library Python backend.
External libraries are separate from proprietary code; they can ship upstream
binary wheels, but normal installation must require no end-user compiler or
manual build configuration. Keep Optuna and the native evolving pipeline intact.
Never fake dependency metadata or describe the entire dependency graph as pure.

Follow docs/ARCHITECTURE_1_0.md, docs/Relazione_MorphTree.txt and the current
qualification matrix. Preserve one-parent diagnosis, executable mutations,
justified acceptance/rollback, continuation, freeze and final numeric Optuna.
Retain valid and negative evidence. No new training without a finite approved
scope; no commit, remote write, tag or publication without owner authorization.
Keep release NO_GO until its mandatory gates pass.

## Product completion direction — 26 October 2026

Preserve the qualified Python engine. Both task facades default to adaptive
structural evolution and use_optuna=True; False disables only final numeric
optimization. Automatic defaults are finite:150 seconds, one worker, two
mutation opportunities and two numeric trials. Keep truthful skip/error/budget
reporting; never claim a numerical phase ran merely because it was enabled.
Consolidate declared capabilities and model reuse. Additional mutation families,
causal claims and broad performance work belong after the reliable2.0 release.
Python3.12–3.15 remains the objective; an external ordinary315 installation
blocker requires an explicit owner release-scope decision, never a silent
support reduction. Authentic unavailable systems remain unqualified. No new
training, Git write or publication is authorized by product consolidation.

## Approved first-release scope — 26 October 2026 (supersedes315 release gate)

The owner explicitly selected option2. Prepare2.0.0 for Windows AMD64 CPython
3.12–3.14 and Ubuntu24.04 WSL2 x86_64 CPython3.12 only. Python3.15 stays
experimental and on the cumulative roadmap; it no longer blocks this scoped
first release. Keep all functional, Pure-Python and safety requirements.
Develop/verify in PRIVATE_CODEX before curated public integration. The owner
copies the prepared public checkout online. No commit, push, tag, publication,
remote operation or further training is authorized. Local technical readiness
must be reported for final owner approval; it is never publication permission.

## Definitive RC approved — 26 October 2026

The owner approved2.0.0 as the definitive release candidate for manual transfer
to the official repository, within>=3.12,<3.15 and the qualified OS profiles.
Local technical preparation is accepted. Preserve every result, manifest and
negative evidence. The owner alone transfers files. CI on the final commit and
TestPyPI verification follow that transfer; PyPI needs separate later approval.
No agent commit, push, tag, remote operation or publication is authorized.
Keep introspection, native mutations, precision, speed and experimental315 on
the cumulative roadmap. This approval grants no new training or automatic work.
