# Migration to native MorphTree2.1.0

The owner definitively selected2.1.0, superseding the temporary3.0.0rc1 label.
Historical Python-only2.1.0 was placed onGitHub,notPyPI according to the owner.
These implementations remain distinguishable by paths,source inventories and
artifact hashes. Install this exact wheel into a FRESH environment;pip can
otherwise regard a different local2.1.0 installation as already satisfied.
Never overwrite an already published distribution filename with different bytes.

Public API,models2.0/2.1,learned state,integrity validation and explicit capacity8
remain available. Training exports are tied to an exact runtime manifest and
must be regenerated after a runtime change;standalone inference is separate.
The internal native-core/model schema version is distinct from packageversion.
Preserve existing GitHubv2.1.0 tags/releases.If they reference oldPython code,
do not move/delete them:the owner may choose a distinct tag such asnative-v2.1.0
for the new commit,while the package remains2.1.0. No tag is created here.
