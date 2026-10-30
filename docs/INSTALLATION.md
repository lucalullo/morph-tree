# Installation and development

Use Python>=3.12,<3.15 on a qualified x86_64 platform. This candidate is not
published. Create a fresh venv and use its interpreter for all commands:

Windows PowerShell (choose an absolute path to the supplied Windows wheel):

```powershell
py -3.12 -m venv .morphtree-rc
& .\.morphtree-rc\Scripts\python.exe -m pip install C:\path\to\morphtree-2.1.0-py3-none-win_amd64.whl
& .\.morphtree-rc\Scripts\python.exe -m pip check
& .\.morphtree-rc\Scripts\python.exe -I -c "import morphtree; from morphtree._native.loader import runtime_info; print(morphtree.__version__); print(runtime_info())"
```

Ubuntu24.04 WSL2 x86_64 (use the supplied ABI-audited manylinux wheel):

```sh
python3.12 -m venv .morphtree-rc
.morphtree-rc/bin/python -m pip install /path/to/morphtree-2.1.0-py3-none-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl
.morphtree-rc/bin/python -m pip check
.morphtree-rc/bin/python -I -c 'import morphtree; from morphtree._native.loader import runtime_info; print(morphtree.__version__); print(runtime_info())'
```

The venv interpreter is env/Scripts/python on Windows or env/bin/python on Linux.
Optuna5.0.0 is mandatory; pandas/sklearn are optional. Compatible wheel
installation, import and inference require no compiler and perform no compilation.

Source builds require MSVC x64 plus Windows SDK on Windows, or a C11 compiler,
linker and libc development headers on Linux (Ubuntu build-essential).
Build scripts discover installed tools, never install them. Windows uses the
static CRT; Linux qualification is Ubuntu24.04WSL2/glibc2.39.
The delivered manylinux tags are ABI-audited; they do not extend whole-pipeline
runtime qualification beyond the environments actually tested.

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

## Linux binary distribution audit
Auditwheel6.8.2 found onlyGLIBC_2.2.5 versioned symbols fromlibc and no extra
shared libraries to bundle. The repaired wheel carries manylinux platform tags,
including manylinux2014/manylinux_2_17. File-by-file validation confirms identical
MorphTree code and native ELF bytes:only WHEEL/RECORD container metadata changes.
This is a binary ABI/symbol/ISA audit,not a new complete-pipeline qualification
on older glibc or other Linux distributions. Actual runtime tests remain the
documented Ubuntu24.04WSL2x86_64/Python3.12–3.14 profiles. No compiler is needed
when installing the compatible precompiled wheel.

Linux builders can install the development-only tools auditwheel==6.8.2,
pyelftools==0.33,packaging==26.3,patchelf==0.19.1.0 into an isolated environment,
along with build==1.3.0/hatchling==1.27.0,then run:

    python -I -B tools/build_artifacts.py --manylinux

The build retains its raw wheel and audit logs in its owned temporary copy.
It refuses repair if any code/library/other file payload changes;no new runtime
dependency is added. Standard source builds still require a local C toolchain.
