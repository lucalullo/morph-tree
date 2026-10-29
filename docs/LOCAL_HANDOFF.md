# Manual transfer — pending final approval

This staging folder updates the existing MorphTree project. It has no .git.
Local functional qualification passes within docs/QUALIFICATION.md. Version 2.1.0
is owner-approved. Publication remains NO_GO pending the post-transfer gates.

After owner approval:
1. Back up the destination checkout and review existing local changes.
2. Verify this folder using tools/check_repository.py.
3. Review docs/FILE_CHANGES.json. Copy added/changed files with their paths.
   Remove only explicitly reviewed obsolete files. In particular, the old root
   morphtree/ moves to python/morphtree/: stale copies risk wrong imports.
   Never overwrite or delete .git.
4. Verify the copied inventory, build wheel/sdist and test isolated installations.
5. Review the Git diff, license, metadata, dependency and platform claims.
6. The owner alone commits, pushes, tags and runs hosted CI on the final commit.
   TestPyPI verification precedes a separate PyPI decision.

No laboratory evidence, private data, personal paths, venvs, compiler scratch
files or dependency wheelhouses belong in this checkout. Manual workflows are
prepared, not executed. The NO_GO publication gate remains active even though the
version has been approved. No credentials are needed for local delivery checks.

## Version and artifact checklist

The transition from the historical experimental `2.1.0+native.lab` identifier to
the owner-approved `2.1.0` changes content and requires actual rebuilds:

- Align pyproject.toml, package `__version__`, API `PUBLIC_USER_RUNTIME`, changelog,
  README and docs/RELEASE_STATUS.json. The build helper reads the declared version
  and rejects disagreement with the Python/API constants.
- Preserve historical provenance and its hashes; the separate Python-only 2.1.0
  reference must never be overwritten or identified by version number alone.
- Regenerate package/source manifests, docs/FILE_CHANGES.json and SOURCE_MANIFEST.
  Build fresh Windows/Linux wheel and sdist artifacts; compute new SHA256 values.
- Check fresh environments, installed metadata/native library identity, two
  consecutive zero-fit test passes, retained-model inference and persistence.
- Keep workflow_dispatch manual and publication blocked. Only the owner may
  authorize hosted CI, reviewed final hashes, TestPyPI, then a separate PyPI gate.

This candidate performs that transition locally. No old distribution is renamed
or reused as proof of a new artifact. The final source manifest and artifact
manifest are the handoff identities. Zero-fit CI does not prove a fresh training
run; the finite historical LIVE evidence is separately documented.

## Trasferimento manuale nel repository GitHub esistente (italiano)

Questo archivio contiene **solo il contenuto del repository** e include `.github/`.
Non contiene una cartella `.git/`; la storia del repository esistente deve
rimanere sotto il controllo del proprietario. La versione e **2.1.0**.

1. Conserva un backup indipendente dell'attuale checkout Git e verifica che il
   suo lavoro locale sia pulito o annotato. Non cancellare i vecchi tag Git.
2. Estrai lo ZIP. La cartella da copiare e `morph-tree/`, non la directory che
   contiene lo ZIP o un ulteriore livello parent.
3. Prima di copiare, dal `morph-tree/` estratto esegui:

   ```powershell
   python -I -B tools/check_repository.py
   ```

4. In PowerShell imposta **manualmente** i percorsi veri, senza incollare
   directory di esempio come se fossero reali:

   ```powershell
   $Sorgente = 'PERCORSO\morph-tree'
   $Destinazione = 'PERCORSO_DEL_TUO_CHECKOUT_GIT\morph-tree'
   if (-not (Test-Path -LiteralPath (Join-Path $Destinazione '.git'))) {
       throw 'La destinazione non contiene .git: verifica il checkout prima di copiare.'
   }
   robocopy $Sorgente $Destinazione /E /XD .git /R:1 /W:1
   ```

   `robocopy` restituisce spesso codici da 0 a 7 anche quando la copia riesce;
   un codice 8 o superiore indica errori. **Non utilizzare `/MIR` o `/PURGE`**:
   potrebbero cancellare file della destinazione, inclusi dati non previsti.
5. Consulta `docs/FILE_CHANGES.json`. Rimuovi *solo dopo una revisione manuale*
   i file obsoleti della versione precedente. In particolare, la vecchia
   directory sorgente di primo livello `morphtree/` non deve prevalere su
   `python/morphtree/`; verifica prima eventuali modifiche personali.
6. Dalla radice del checkout di destinazione esegui:

   ```powershell
   python -I -B tools/check_repository.py
   git status --short
   git diff --check
   ```

7. Verifica build e installazioni in ambienti isolati e, **solo dopo aver
   approvato le differenze**, esegui personalmente commit, push e workflow CI.
   Prima TestPyPI, poi eventuale PyPI con una nuova autorizzazione.

Non sovrascrivere mai `.git`, non pubblicare credenziali o dati privati e non
interpretare il GO locale come approvazione alla pubblicazione.


## Consistency with the NeuroTabular repository

The public repository uses analogous contributor-facing files (issue templates,
PR template, code of conduct, release checklist, publishing and versioning
documentation) and a user-facing README. Workflow details are **not copied**
between projects: MorphTree distributes platform-specific compiled kernels and
retains its existing manually gated CI/publication process. No NeuroTabular
accounts, tokens, runtime dependencies or PyTorch files are used here.
