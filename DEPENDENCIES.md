## Current2.0.0 preparation

The proprietary engine is entirely Python. The default public trainer uses
the standard-library backend and ships no proprietary native/JIT extension.
This does not mean that every dependency is Pure Python: NumPy is required by
Optuna; PyYAML, SQLAlchemy, MarkupSafe and optional pandas may include upstream
binary extensions. Compatible official wheels avoid user compilation; the
unresolved ordinary Windows3.15 PyYAML source-build path blocks official3.15
support, deferred by the owner beyond the first2.0 release.

Official Python fallbacks were checked for safe YAML parsing, SQLAlchemy date
processing and MarkupSafe escaping. The sampled outputs agree. Five-sample
microchecks with Python allocation tracing measured, respectively, roughly
31.40/2.83ms for three Python/C YAML parses,0.210/0.306ms for300 date conversions
and1.264/0.382ms for500 HTML escapes. These are narrow function measurements,
not training performance, full fallback qualification or aggregate RSS.
No installed dependency mode, feature or safety control was changed.
There is no qualified Pure-Python NumPy drop-in in this work; removing it would
break the declared Optuna contract. No dependency replacement is silently shipped.

See [the tested2.0 profiles](docs/TESTED_DEPENDENCIES_2_0.json) and
[release gates](docs/RELEASE_QUALIFICATION.md). Windows3.12–3.15 native LIVE
profiles passed, with3.15 in a prepared official Pure-PyYAML laboratory venv.
Normal fresh public-index315 installation remains blocked. No system compiler
is installed or required for the completed laboratory qualification; the
end-user compiler/manual-configuration-free goal is unchanged. The historical
dependency snapshot below is preserved separately, not relabeled as2.0 proof.

# Dependencies and licensing

MorphTree's own code remains under the existing MIT license. No vendor learner
is bundled as an internal engine. Optuna==5.0.0 is a standard dependency and its
transitive dependencies carry their own terms;MorphTree's MIT license does not
relicense them. Hatchling==1.27.0 is the declared build backend.

No dependency binaries,virtualenvs or private wheelhouse are included here.
The historical qualification used RECORD-verified repacked archives with retained
license files. Later ordinary official-index installations and dependency-file
comparisons are recorded separately. Final offline verification reuses those
qualified contents to avoid remote operations; it is not a new index fetch.
No blanket vulnerability absence or upstream identity for repacked archives is claimed. Consult each actually installed
distribution's METADATA/license files before redistribution. Unknown metadata
must remain unknown. The tested environment is Windows x64/CPython3.12.

The dependencies installed during qualification are documented in the public
research/completion report. Build/install normally through declared metadata;
do not copy site-packages into the package or add a research checkout to PYTHONPATH.

The exact tested distribution versions,declared license fields (UNKNOWN/missing
remain unknown),license file hashes and Python constraints are recorded in
[TESTED_DEPENDENCIES.json](docs/TESTED_DEPENDENCIES.json). This is a metadata
snapshot,not an index-install hash lock or a blanket license/security assertion.
