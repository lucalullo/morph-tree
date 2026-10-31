# Installation

## Standard installation

```bash
pip install morphtree
```

This selects the latest compatible release available on PyPI. To select this
release explicitly, use `python -m pip install morphtree==2.1.0` once its
compatible artifacts are available. Index availability is a separate release
check; local package validation does not establish publication.

Use CPython 3.12, 3.13 or 3.14 on a supported x86-64 Windows or Linux platform.
A compatible prebuilt wheel includes the native C library: no C compiler,
manual backend configuration or runtime compilation is required. Optuna 5.0.0
and NumPy are installed as core dependencies. pandas and scikit-learn are
optional; the README dataset examples use both.

An isolated environment is recommended:

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install morphtree
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -I -c "import morphtree; print(morphtree.__version__)"
```

Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install morphtree
.venv/bin/python -m pip check
.venv/bin/python -I -c 'import morphtree; print(morphtree.__version__)'
```

Use the selected environment interpreter for every subsequent command. Install
optional example dependencies with `python -m pip install scikit-learn pandas`.
See [the API](API.md) for fit, NumPy predictions, persistence and exports.

## Install a reviewed local artifact

For offline review or before publication, substitute the absolute wheel path:

```text
python -m pip install /absolute/path/to/morphtree-2.1.0-<compatible-tags>.whl
python -m pip check
python -I -c "from morphtree._native.loader import runtime_info; print(runtime_info())"
```

Replace the placeholder with the actual wheel filename. Windows wheels use
`win_amd64`; delivered Linux wheels carry audited manylinux x86-64 tags. Check
the delivery SHA-256 before installation. Use a fresh environment to isolate dependencies and verify the
installed artifact against the reviewed delivery manifest.

## Build from source and develop

Installing from a source distribution or source checkout requires an existing
C toolchain: MSVC x64 plus Windows SDK on Windows, or a C11 compiler, linker and
libc development headers on Linux (for example Ubuntu build-essential).
Build scripts discover tools but never install them. Windows uses the static
CRT. These development requirements do not apply to compatible prebuilt wheels.

```text
python -m pip install build==1.3.0 hatchling==1.27.0
python -I -B tools/check_repository.py
python -I -B tools/build_artifacts.py
```

Builds use a separate source copy and write wheel and sdist to `dist/`. Install
each artifact in its own clean environment. The sdist includes native sources
and build tools. After installation, use that interpreter to execute
`python -I -B tools/check_installed.py` from the source checkout. The checker
copies tests to a separate directory, prohibits learner/tree fits and repeats
the suite; it is a software check, not a new training benchmark.

## Linux wheel audit and runtime qualification

The delivered Linux wheel is audited with auditwheel 6.8.2. Its C library uses
only GLIBC_2.2.5 versioned symbols from libc and requires no additional shared
libraries. Repair adds compatible manylinux tags, including manylinux2014 and
manylinux_2_17, while checking that code and ELF payloads remain identical.
This ABI/symbol/ISA audit does not establish whole-pipeline execution on every
older glibc distribution. Runtime qualification is Windows x64 and Ubuntu
24.04 WSL2 x86-64/glibc 2.39 with CPython 3.12–3.14; see [qualification](QUALIFICATION.md).
Python 3.15, macOS and other architectures are not in that verified scope.

Linux builders can install development-only auditwheel==6.8.2,
pyelftools==0.33, packaging==26.3 and patchelf==0.19.1.0 in an isolated build
environment, then run:

```text
python -I -B tools/build_artifacts.py --manylinux
```

Raw wheels and audit logs are retained in the owned build copy. Repair is
rejected if any package or library payload changes. Final hosted CI and
package-index installation are maintainer release checks, separate from
local builds and historical runtime qualification.
