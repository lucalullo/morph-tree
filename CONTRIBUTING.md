# Contributing to MorphTree

MorphTree is an experimental native tree-learning project. Please report bugs with a minimal reproducer, task, Python/OS versions, expected/observed behavior and available logs **without** personal data, private research or credentials.

The default GitHub Actions job is **zero-fit only**: it checks packaging, source integrity and public contracts, not predictive quality. Do not run or publish full research experiments without a specified data protocol, budget, applicable permissions and independent validation. See [documentation](docs/RELEASE_QUALIFICATION.md) and [security](SECURITY.md).

Source changes must update the [local source manifest](LOCAL_SOURCE_MANIFEST.json) and pass `python -I -B tools/check_repository.py`. The first release is scoped to Windows x64/Python 3.12; other platforms should not be described as qualified without evidence. No package index or GitHub release publication is authorized by a code contribution.
