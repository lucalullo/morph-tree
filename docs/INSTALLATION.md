# Installing MorphTree2.0.0

The owner-approved first-release scope is Windows AMD64/CPython3.12–3.14 and
Ubuntu24.04 WSL2 x86_64/CPython3.12. Exact native profiles: Windows3.12.10,
3.13.17,3.14.9 and Ubuntu3.12.3. Python3.15 remains experimental and is excluded
by `Requires-Python: >=3.12,<3.15`. This Python range cannot encode OS-specific
qualification; Linux3.13/3.14 and macOS are not supported claims.

From the prepared repository, or after receiving its verified wheel/sdist:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install .
.venv\Scripts\python -m pip check
.venv\Scripts\python -c "import morphtree; print(morphtree.__version__)"
```

The installed version should read `2.0.0`. On Linux use
`python3.12 -m venv .venv` and `.venv/bin/python -m pip install .`. Optuna `5.0.0` remains required. Use
`MorphTreeClassifier(use_optuna=False)` or `MorphTreeRegressor(use_optuna=False)`
to disable only numerical tuning. Both use the same native evolutionary engine.

For a local build, replace `.` with the verified `morphtree-2.0.0-py3-none-any.whl`
or `morphtree-2.0.0.tar.gz` path. No index upload is authorized by this work.
Do not use an index version as a substitute for a SHA-256-identified local
artifact. The complete [Python 3.15 blocker and system matrix](RELEASE_BLOCKERS_2_0.md)
distinguish the successful Pure-Python laboratory from the unresolved ordinary
public-index3.15 install. That future-support issue does not block the scoped
first release. No compiler or special PyYAML setting is needed by supported
profiles. Optuna remains enabled by default. No release has been uploaded here.

## Runtime limits

Fresh Windows qualification environments on this machine encountered an
application-control error loading official pandas DLLs. This is retained as an
environment-specific failure; no security policy was disabled. pandas remains
optional: separate mandatory-dependency profiles pass the documented core
NaN/categorical and model-inference checks. The earlier named pandas profiles
remain recorded; a new full pandas check in these temporary environments is
still blocked. See the current qualification matrix before making support claims.

Defaults: 150-second cooperative global budget, one worker, two initial stages/members, two structural attempts and two final Optuna trials. The library may return an earlier verified model if a soft budget cutoff prevents a complete search; inspect the report. Hard failure conditions are errors, not success. No absolute in-process peak-memory guarantee exists.

See [the task API](API_2_0.md), [legacy API](API_REFERENCE_1_0.md),
[security](../SECURITY.md) and [release verification](RELEASE_QUALIFICATION.md).

## WSL storage

Keep model files and `load(..., workdir=...)` on the native Linux filesystem
when using WSL. A paired zero-fit check of the same saved models, interpreter
and input found a substantial inference penalty under mapped NTFS (`/mnt/c`)
with integrity checks preserved. This observation does not isolate the cause
of training-time differences or establish a general speedup. Training evidence
is limited to the exact named qualified profile.
