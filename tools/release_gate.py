"""Strict checks for MorphTree's manual TestPyPI -> PyPI artifact promotion.
No publishing or network access occurs unless --require-ci or --verify-origin is explicitly passed.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'
TAG = 'v1.0.0'
WHEEL = 'morphtree-1.0.0-py3-none-any.whl'
SDIST = 'morphtree-1.0.0.tar.gz'
WORKFLOW = 'publish.yml'
ARTIFACT_NAME = 'morphtree-1.0.0-testpypi-approved'


def check_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gh_api(path):
    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    if not token or not repo or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Missing GitHub identity or credential')
    request = urllib.request.Request('https://api.github.com/repos/' + repo + '/' + path,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'morphtree-release-gate'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def git(args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, timeout=20).strip()


def preflight(require_ci=False):
    from importlib import metadata
    sha = os.environ.get('GITHUB_SHA', '')
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('Expected exact GitHub commit SHA')
    if git(['rev-parse', 'HEAD']) != sha:
        raise ValueError('Workflow checkout not at approved SHA')
    if git(['rev-parse', f'refs/tags/{TAG}^{{commit}}']) != sha:
        raise ValueError('Tag does not point to the workflow commit')
    if not (ROOT / 'pyproject.toml').read_text('utf8').count('version="1.0.0"') == 1:
        raise ValueError('Release metadata differs from 1.0.0')
    if os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch':
        raise ValueError('Manual workflow required')
    if os.environ.get('GITHUB_REF_NAME') not in ('main', TAG):
        raise ValueError('Unexpected ref; select main or the v1.0.0 tag')
    subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'tools/check_repository.py'),
                    '--git-head', '--expected-commit', sha], check=True, timeout=50)
    if require_ci:
        result = gh_api('actions/workflows/test.yml/runs?head_sha=' + sha + '&event=workflow_dispatch&per_page=100')
        success = [x for x in result.get('workflow_runs', [])
                   if x.get('head_sha') == sha and x.get('conclusion') == 'success']
        if not success:
            raise ValueError('No successful hosted CI run for the exact release commit')
    print('PASS_RELEASE_PREFLIGHT', sha, TAG)


def list_artifacts(base):
    dist = Path(base) / 'dist'
    names = sorted(x.name for x in dist.iterdir() if x.is_file()) if dist.is_dir() else []
    if names != sorted((WHEEL, SDIST)):
        raise ValueError('Distribution directory must contain only the expected wheel and sdist: ' + repr(names))
    return {name: {'sha256': check_hash(dist / name), 'bytes': (dist / name).stat().st_size}
            for name in (WHEEL, SDIST)}


def make_manifest(base):
    base = Path(base)
    artifacts = list_artifacts(base)
    run_id = os.environ.get('GITHUB_RUN_ID', '')
    if not run_id.isdigit():
        raise ValueError('Missing release run ID')
    data = {'schema': 'morphtree-artifact-provenance-v1', 'package': 'morphtree',
            'version': VERSION, 'tag': TAG, 'sha': os.environ['GITHUB_SHA'],
            'repository': os.environ['GITHUB_REPOSITORY'], 'origin_run_id': int(run_id),
            'origin_target': 'testpypi', 'artifacts': artifacts}
    target = base / 'release-meta' / 'provenance.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n', encoding='utf8')
    print('PASS_ARTIFACT_MANIFEST', json.dumps(artifacts, sort_keys=True))


def verify(base, expected_run_id=None, remote=False):
    base = Path(base)
    meta = json.loads((base / 'release-meta' / 'provenance.json').read_text('utf8'))
    if meta.get('schema') != 'morphtree-artifact-provenance-v1' or meta.get('version') != VERSION:
        raise ValueError('Unexpected artifact manifest')
    if meta.get('tag') != TAG or meta.get('sha') != os.environ.get('GITHUB_SHA'):
        raise ValueError('Manifest tag/commit mismatch')
    if meta.get('repository') != os.environ.get('GITHUB_REPOSITORY') or meta.get('origin_target') != 'testpypi':
        raise ValueError('Artifact origin mismatch')
    if not isinstance(meta.get('origin_run_id'), int) or meta['origin_run_id'] <= 0:
        raise ValueError('Invalid release run')
    if expected_run_id is not None and meta['origin_run_id'] != int(expected_run_id):
        raise ValueError('Different TestPyPI origin run')
    if meta.get('artifacts') != list_artifacts(base):
        raise ValueError('Artifact content/hash mismatch')
    if remote:
        run = gh_api('actions/runs/' + str(meta['origin_run_id']))
        if (run.get('head_sha') != meta['sha'] or run.get('conclusion') != 'success'
                or run.get('event') != 'workflow_dispatch'
                or WORKFLOW not in run.get('path', '')):
            raise ValueError('TestPyPI source workflow not successful or mismatched')
        # Origin marker is written only by the testpypi build branch. Check manifest
        # plus successful origin run, never rebuild wheel/sdist in promotion.
    print('PASS_REPRODUCIBLE_ARTIFACTS', meta['origin_run_id'])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('preflight'); q.add_argument('--require-ci', action='store_true')
    q = sub.add_parser('make-manifest'); q.add_argument('folder')
    q = sub.add_parser('verify'); q.add_argument('folder'); q.add_argument('--origin-run-id'); q.add_argument('--require-origin-success', action='store_true')
    args = p.parse_args()
    if args.cmd == 'preflight':
        preflight(args.require_ci)
    elif args.cmd == 'make-manifest':
        make_manifest(args.folder)
    else:
        verify(args.folder, args.origin_run_id, args.require_origin_success)

if __name__ == '__main__':
    main()
