# MorphTree 1.0.0

**An experimental approach to tabular machine learning through autonomous, validation-guided evolution of native tree-learning algorithms.**

MorphTree is an original Python project for **binary classification, multiclass classification and regression**. Its standard `MorphTree().fit(X, y)` compares qualified native families, selects one parent and tests executable structural changes to its learning rules. Validated improvements may be accepted; other changes are rolled back. The search stops within a finite budget, freezes the resulting program and optionally applies final numerical Optuna tuning.

```python
from morphtree import MorphTree
model = MorphTree(random_state=42, verbosity=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

For classification, use `model.predict_proba(X_test)`. Explicit `task="binary"`, `task="multiclass"` and `task="regression"` resolve ambiguous targets. `use_optuna=False` skips only final numerical tuning. Models can be saved/loaded; program, recipe and mutation history can be inspected; standalone export covers inference only.

**Supported qualification environment:** Windows x64 with CPython 3.12. **Requirements:** `optuna==5.0.0`. The engine is native Pure Python; it does not use external vendor ML estimators. The 1.0 portfolio is native boosting and bagging with a tested depthwise structural transformation. Current children require retraining. Resource limits are finite but partly cooperative; no hard RSS bound or superiority over mature alternatives is claimed. Improvements are not guaranteed.

Evidence from small authorized LIVE campaigns supports the three-task execution and correct rollback. Independent predictive benefit from automatically accepted mutations has not been demonstrated. Read the public [README](https://github.com/lucalullo/morphtree) once the repository is published, or use the local `README.md`, `docs/` and `research/` bundled with the source repository. Historical research and prerelease candidate identifiers are development checkpoints, not previous public releases.

MIT License, copyright 2026 Luca Lullo.
