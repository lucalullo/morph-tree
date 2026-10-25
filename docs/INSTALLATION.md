# Installation - MorphTree 1.0.0

The qualified environment is **Windows x64 with CPython 3.12**. The code is Pure Python; other environments are not yet part of the verified support contract.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install .
.venv\Scripts\python -m pip check
.venv\Scripts\python -c "import morphtree; print(morphtree.__version__)"
```

The version should read `1.0.0`. Optuna `5.0.0` is pinned as a required dependency for the default numerical finalization. Use `MorphTree(use_optuna=False)` to disable only numerical tuning. The model-training engine is native and does not import competitor learners as internal estimators.

For a locally built distribution, replace `.` with the specific `morphtree-1.0.0-py3-none-any.whl` or `morphtree-1.0.0.tar.gz` path. Once PyPI publication is independently completed, `python -m pip install morphtree==1.0.0` becomes appropriate. **Do not treat an unrelated index package as this checked build without verifying its identity.**

## Runtime limits

Defaults: 150-second cooperative global budget, one worker, two initial stages/members, two structural attempts and two final Optuna trials. The library may return an earlier verified model if a soft budget cutoff prevents a complete search; inspect the report. Hard failure conditions are errors, not success. No absolute in-process peak-memory guarantee exists.

See [the API](API_REFERENCE_1_0.md), [security](../SECURITY.md) and [release verification](RELEASE_QUALIFICATION.md).
