## Change summary

Explain the problem and the smallest justified correction.

## Compatibility and scientific behavior

Describe effects on public API, saved models, learning program, structural mutation
history, acceptance/rollback, numerical outcomes, native ABI, dependencies and
supported platforms. Write "none" where applicable.

## Verification

- [ ] `python -I -B tools/check_repository.py` passes **after regenerating the inventory** for this reviewed change.
- [ ] Relevant zero-fit tests and repeat-run tests pass.
- [ ] Native changes include bounds/ABI tests and appropriate build evidence.
- [ ] Algorithm or performance changes include a preregistered comparison, budget and maintainer approval before LIVE fits.
- [ ] User-visible changes update README, CHANGELOG and API/qualification docs as necessary.
- [ ] No credentials, personal paths, private results, `.git`, compiled scratch files or datasets are included.
- [ ] Wheel/sdist and installed-package qualification are repeated when release content changes.

Git history, merges, versioning and all publication actions remain owner-controlled.
