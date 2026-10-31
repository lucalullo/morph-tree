# Manual transfer and later publication — owner actions only
Use this entire repository including.github,excluding.git and local build/cache
directories. Preserve the destination.git. The supplied obsolete-file inventory
lists old tracked-layout files to remove manually after keeping a backup; old
top-level morphtree/ must not shadow the new python/morphtree/ package layout.
Do not copy private backups,models,datasets,raw results or environments.

1.Review source/artifactSHA256 and remaining qualification gates.
2.Transfer locally,then the owner performs all Git operations personally.
3.Run hosted build/test on the exact final commit; retain wheel/sdist hashes.
4.Only after separate approval,owner uploads the reviewed2.1.0 artifacts toTestPyPI,creates
fresh environments,installs the exact version,checks native loading,three tasks,
API,persistence,exports and historical models. Local checks do not replace this.
5.PyPI requires separate final approval after TestPyPI validation;never rebuild
or overwrite approved filenames silently. A final version needs its own
metadata/artifact rebuild and checks. No publication is executed here.

The publication workflow is manual and fails closed while RELEASE_STATUS.json
has publication_approved=false. No token is included or configured by this kit.

The current Linux delivery includes an auditwheel-verified manylinux wheel with
unchanged code and ELF library bytes. Whole-pipeline qualification remains the
actual Ubuntu24.04WSL2 profiles;do not claim new older-Linux runtime tests.
The owner still performs final hostedCI and TestPyPI before any PyPI approval.

## Package version2.1.0 and existing GitHub labels
The current package is2.1.0. A GitHub tag/release and aPyPI release are distinct.
Check the public index/tag reports before release. Preserve any existingv2.1.0
ref/release unchanged;use a new distinct label only by owner decision,not a
force-update or historical asset replacement. Local preparation performs none.
