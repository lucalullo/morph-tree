# Migration to the native MorphTree 2.1.0 candidate

Python imports, fit/predict, random_state=None, n_jobs='auto', validated workdir
loading, stop_reason_, recipe/program/report/genealogy and exports retain their
contracts. Compatible wheels now supply the C backend automatically; building
from source requires a C toolchain.

Retained2.0/2.1JSON models remain readable without rewriting or retraining weights.
Historical models keep their decoder. Use the same OS/interpreter and fixed seed
for exact reproducibility. Training fragments bind an exact runtime manifest;
regenerate them from a verified model after a runtime change. Standalone
predictors are a distinct export.

The owner selected 2.1.0 for this native candidate, superseding the former 3.0.0
proposal. The separate Python-only reference also bears 2.1.0 locally; this does
not make their package contents interchangeable. Compare source manifests, wheel
hashes, `_experimental/manifest.json` and the loaded native binary identity.
Install into a fresh environment to avoid pip treating an existing, different
2.1.0 installation as already satisfied. The local build/qualification workflow
always uses fresh environments and explicit artifacts. Do not overwrite any
previously published distribution with different bytes under the same filename.

The previously published 2.0.0 remains unchanged. Native 2.1.0 is prepared locally;
publication still requires the owner's gates in [LOCAL_HANDOFF.md](LOCAL_HANDOFF.md).
