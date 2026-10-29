# Versioning and compatibility

MorphTree's native Python+C distribution is identified as **2.1.0** by the
project owner. This is a **source and local-qualification identity**, not proof
that version 2.1.0 has been uploaded to any package index.

Older published versions and Git tags are immutable historical identities.
Never replace an already published distribution with different files under the
same version. Before releasing, verify whether this exact distribution version
is available on the chosen index and use a new, owner-approved version if there
is a collision. Python-only and native references that share a local version
string are distinguished by their immutable source and artifact SHA-256 hashes.

The public classes remain `MorphTreeClassifier` and `MorphTreeRegressor`. API,
serialized-model compatibility and the learning-program contract must be tested
independently from version labels. Compatibility claims refer only to verified
formats and platforms: see [qualification](docs/QUALIFICATION.md) and
[migration](docs/MIGRATION.md).

Only the owner chooses the next version and authorizes tag, release and
publication. No version change by itself creates scientific evidence.
