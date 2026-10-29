# Installation and development

Use Python>=3.12,<3.15 on a qualified x86_64 platform. This candidate is not
published. Create a fresh venv and use its interpreter for all commands:

```text
python -m venv env
python -m pip install PATH_TO_COMPATIBLE_WHEEL
python -m pip check
python -m morphtree --help
```

The venv interpreter is env/Scripts/python on Windows or env/bin/python on Linux.
Optuna5.0.0 is mandatory; pandas/sklearn are optional. Compatible wheel
installation, import and inference require no compiler and perform no compilation.

Source builds require MSVC x64 plus Windows SDK on Windows, or a C11 compiler,
linker and libc development headers on Linux (Ubuntu build-essential).
Build scripts discover installed tools, never install them. Windows uses the
static CRT; Linux qualification is Ubuntu24.04WSL2/glibc2.39.
A linux_x86_64 wheel is not a manylinux portability claim.

```text
python -m pip install build==1.3.0 hatchling==1.27.0
python -I -B tools/check_repository.py
python -I -B tools/build_artifacts.py
```

Builds use a separate copy and write wheel/sdist to dist/. Install each artifact
in a separate clean venv. Using that installed interpreter run
`python -I -B tools/check_installed.py`. Tests are copied outside the repository;
learner/tree fits are forbidden. Unit weights are not LIVE scientific evidence.
The sdist includes native sources and build tools. Hosted workflows are manual
owner actions, not evidence of hosted CI success.
