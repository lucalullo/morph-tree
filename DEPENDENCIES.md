# Dependencies and licensing

MorphTree's own code remains under the existing MIT license. No vendor learner
is bundled as an internal engine. Optuna==5.0.0 is a standard dependency and its
transitive dependencies carry their own terms;MorphTree's MIT license does not
relicense them. Hatchling==1.27.0 is the declared build backend.

No dependency binaries,virtualenvs or private wheelhouse are included here.
Local qualification used RECORD-verified repacked archives with retained license
files,not fresh original upstream archives. Normal future index resolution and
advisory/provenance checks require their own environment/approval;no vulnerability
absence or upstream archive identity is claimed. Consult each actually installed
distribution's METADATA/license files before redistribution. Unknown metadata
must remain unknown. The tested environment is Windows x64/CPython3.12.

The dependencies installed during qualification are documented in the public
research/completion report. Build/install normally through declared metadata;
do not copy site-packages into the package or add a research checkout to PYTHONPATH.

The exact tested distribution versions,declared license fields (UNKNOWN/missing
remain unknown),license file hashes and Python constraints are recorded in
[TESTED_DEPENDENCIES.json](docs/TESTED_DEPENDENCIES.json). This is a metadata
snapshot,not an index-install hash lock or a blanket license/security assertion.
