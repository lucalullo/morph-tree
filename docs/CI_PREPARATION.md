# MorphTree 1.0.0 - manual GitHub Actions verification

**First upload:** create an empty private GitHub repository and upload the **contents** of the top-level `MorphTree/` folder. Include no auto-generated README/license, previous Git history, private research or untracked artifacts. Initialize and commit Git locally after `python -I -B tools/check_repository.py` reports `PASS_LOCAL_SOURCE_INTEGRITY`. Check `python -I -B tools/check_repository.py --git-head` against the committed snapshot before pushing personally.

**Qualification:** the workflow `.github/workflows/test.yml` is triggered **manually only**, with an explicit `confirm_zero_fit_scope` checkbox. It uses `permissions: contents: read`, one Windows/Python 3.12 worker, a 20-minute job timeout, no release publishing and **no native model training**. The worker verifies the exact committed source, builds wheel/sdist, installs both into separate clean environments and runs installed **zero-fit** package and transport checks.

Review repository Actions permissions, billing/usage controls and workflow-log retention (the YAML cannot enforce account limits). Run GitHub Actions on the exact intended commit. **The workflow has not yet been executed in this chat**; local YAML and test checks are not a hosted CI PASS. Save logs before retention expiry. If it fails, investigate the exact cause before manually rerunning.

A passing hosted zero-fit job does not replace the existing scoped LIVE training evidence or grant automatic official Release GO. The owner decides separately when to tag or publish 1.0.0.

The public repository contains only a curated research index; the full experiment archive stays outside GitHub.
