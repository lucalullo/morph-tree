# MorphTree 2.1.0 — Python API, integrated C backend

Local native delivery ready for owner review in the documented qualified scope.
Not published; post-transfer hosted CI/TestPyPI remain gates. MorphTree implements
its own tabular learning algorithms; no external vendor learner is the engine.

```python
from morphtree import MorphTreeClassifier, MorphTreeRegressor

model = MorphTreeClassifier(random_state=42)
model.fit(X, y)
predictions = model.predict(test)
probabilities = model.predict_proba(test)  # classification only
model.save("model.json")
restored = MorphTreeClassifier.load("model.json")
```

Use MorphTreeRegressor for continuous targets. Missing values and categoricals
are handled natively. Native-family screening selects one parent; TRAIN-only OOF
diagnosis drives executable structural proposals, trained descendants, justified
acceptance/rollback and lineage continuation. The program is frozen before final
numerical Optuna (enabled by default) and refit on all eligible input rows.

## Explicit higher capacity

```python
model = MorphTreeClassifier(random_state=42, capacity={"n_estimators": 8})
```

This public API already exists in the qualified engine. Default screening and
structural search remain at two rounds; the existing automatic post-freeze
numerical domain is unchanged. Eight can require more time/memory. Historical
stellar VALIDATION improved balanced accuracy from0.879211 to0.920422 at8rounds;
static and evolutionary variants were identical at equal capacity/tuning. This
is not independent-test confirmation or proof that mutations improved accuracy.
No planner or experimental diagnostic rule is integrated.

Verified ordered histogram and cached traversal kernels execute in C; mutable
learning rules remain Python. Small batches use equivalent Python traversal.
Compatible precompiled wheels require no compiler. Source builds need MSVC+
Windows SDK or a Linux C11 toolchain. Input is not silently sampled, and resource
admission never disables evolution to obtain a successful fit.

The public task classes use the scalable columnar path. They do not apply the
legacy 2 MiB payload,65,536 TRAIN-row or128 encoded-feature caps retained in
`morphtree.api.MorphTree`. Admission and execution depend on resource estimates,
observed headroom and time limits; large-data completion is not guaranteed.
See the public API/resource limits guide in the repository documentation.

## Scope and installation

Target CPython>=3.12,<3.15 x86_64 on Windows and Ubuntu24.04WSL2/glibc2.39.
See docs/QUALIFICATION.md for historical code evidence versus fresh RC checks.
The Linux wheel is ABI-audited for manylinux tags;full-pipeline tests remain
scoped to Ubuntu24.04WSL2. Python3.15,macOS,Linux large training and million-row
training are unqualified. No general predictive or full-fit speed superiority.

Create a fresh virtual environment, then install an explicit compatible wheel
with that environment's Python. Optuna5.0.0 is mandatory. The standard-library
list interface needs no pandas/NumPy; those input adapters are optional.

See [installation](https://github.com/lucalullo/morph-tree/blob/main/docs/INSTALLATION.md), [API](https://github.com/lucalullo/morph-tree/blob/main/docs/API.md),
[architecture](https://github.com/lucalullo/morph-tree/blob/main/docs/ARCHITECTURE.md), [qualification](https://github.com/lucalullo/morph-tree/blob/main/docs/QUALIFICATION.md),
[comparisons](https://github.com/lucalullo/morph-tree/blob/main/docs/COMPARISON.md), [migration](https://github.com/lucalullo/morph-tree/blob/main/docs/MIGRATION.md) and
[manual handoff](https://github.com/lucalullo/morph-tree/blob/main/docs/LOCAL_HANDOFF.md). MIT license; package name morphtree.
