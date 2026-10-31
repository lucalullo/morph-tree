# Dependencies

MorphTree combines Python and two proprietary C kernels; no vendor learner.
Optuna==5.0.0 and NumPy>=1.26,<3 are mandatory; final tuning remains on by default.
NumPy supplies the public prediction array boundary. NumPy,
SQLAlchemy, PyYAML and transitive dependencies can contain upstream compiled
components. Official compatible wheels avoid user compilation.
Hatchling==1.27.0 and build==1.3.0 are the verified backend/frontend.
pandas and sklearn are optional; universal sklearn compliance is not claimed.

Source builds need an existing documented compiler; runtime never installs tools.
Local checks use retained official wheels offline without altering metadata.
Python3.15 ordinary installation is outside scope. MIT applies to MorphTree;
upstream licenses remain their own. No dependency binaries/venvs/wheelhouses
are redistributed here. Install success is not a full vulnerability audit.
