# Changelog

## 2.0.0 — local development target, not release-ready

- Add public task-specific Classifier/Regressor facades sharing the existing
  native evolutionary engine; classification resolves binary/multiclass only.
- Preserve legacy MorphTree and bounded declarative model loading; expose old
  low-level learner classes under explicit NativeMorphTree names.
- Normalize pandas.NA without a mandatory pandas import and preserve a valid
  encoder after a failed schema refit. Keep TRAIN-only preprocessing and caps.
- Separate native dispatch from public facade names and add zero-fit contracts.
- Complete the approved three-task native LIVE profiles on Windows
  3.12.10/3.13.17/3.14.9 and the prepared 3.15.1 Pure-PyYAML laboratory profile.
  Preserve exact numerical comparisons and correct structural rollback.
- Apply the explicit owner-approved first-release scope: Windows AMD64/Python
  3.12–3.14 and Ubuntu24.04 WSL2 x86_64/Python3.12. Retain ordinary3.15 installation
  as an experimental future-support blocker, preserving all laboratory evidence.
- Reject positive release decisions with narrow or inconsistent package metadata;
  verify the public task constructors during installed-artifact identity checks.
- Preserve historical 1.0.0 artifacts/tag. All training approvals are consumed;
  further LIVE training needs new approval. No remote publication occurs here.

- Promote the retained split mathematics into the public50-test suite;
  verify ordinary official-index wheel/sdist installs on Windows/Linux3.12
  and retained-model standalone inference on Linux without changing runtime.

- Qualify the bounded three-task native pipeline on Ubuntu24.04WSL2/Python3.12.3:
 39 native fits,12 OOF,6 real trials, exact Windows-reference parity and
 correct structural rollback. Preserve slower observed WSL wall times
 and isolate inference filesystem overhead with zero additional fits.

- Validate actual wheel/sdist release metadata and reject direct manifest or
  verification attempts while qualification remains NO_GO. Add five offline
  regression contracts; ten gate tests pass on the five available profiles.

- Promote independent objective-derivative/metric checks into the60-method source
  suite. Complete fresh external core-only Windows profiles and full Linux3.12
  verification, retaining application-control pandas failures separately.
- Correct the public README link in package metadata; keep all native runtime
  bytes and existing LIVE evidence unchanged.

## 1.0.0 - first public release

- Introduce the MorphTree evolutionary `fit()` interface for binary classification, multiclass classification and regression.
- Provide native boosting and bagging candidates, one-parent selection, error diagnosis, executable depthwise structural mutations, acceptance/rollback, finite search stopping and final optional Optuna numerical tuning.
- Include model persistence, bounded loading, inference export, executable-program inspection, CLI inference and search reports.
- Supply a documented research archive retaining negative, failed and inconclusive experiments and the canonical product specification.
- Target Windows x64 and CPython 3.12 initially; document limits and the manual hosted-CI verification gate.

This is the **first public release**. Earlier experimental labels were local research and packaging checkpoints, not previous public releases. For their preserved details see [historical development changelog](research/history/DEVELOPMENT_CHANGELOG_ORIGINAL.md).
