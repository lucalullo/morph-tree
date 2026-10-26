"""Strict checks for MorphTree's manual TestPyPI -> PyPI artifact promotion.
Never publishes. Network reads require --require-ci or --require-origin-success.
"""
import argparse
from email import policy
from email.parser import BytesParser
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '2.0.0'
TAG = 'v2.0.0'
WHEEL = 'morphtree-2.0.0-py3-none-any.whl'
SDIST = 'morphtree-2.0.0.tar.gz'
WORKFLOW = 'publish.yml'
ARTIFACT_NAME = 'morphtree-2.0.0-testpypi-approved'
REQUIRED_PYTHON_SPEC = '>=3.12,<3.15'
SUPPORTED_PLATFORMS = {'Windows AMD64': ['3.12', '3.13', '3.14'],
                       'Ubuntu24.04 WSL2 x86_64': ['3.12']}


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


def qualification_gate():
    """Offline fail-closed check of the separately maintained release decision."""
    value = json.loads((ROOT / 'docs/QUALIFICATION_2_0.json').read_text('utf8'))
    required = ['3.12', '3.13', '3.14']
    if (value.get('schema') != 'morphtree-release-qualification-v1'
            or value.get('version') != VERSION or value.get('required_windows_python') != required
            or value.get('release_ready') is not True):
        raise ValueError('MorphTree 2.0.0 release qualification is incomplete; publication blocked')
    for version in required:
        entry = value.get('windows', {}).get(version, {})
        if any(entry.get(key) != 'PASS' for key in ('wheel','sdist','LIVE_three_tasks','integrity','normal_fresh_pip_install')):
            raise ValueError('Missing required Windows/Python qualification: ' + version)
    if value.get('critical_open_issues') != []:
        raise ValueError('Critical release issues remain open')
    if value.get('supported_platforms') != SUPPORTED_PLATFORMS or value.get('experimental_python') != ['3.15']:
        raise ValueError('Declared platform scope differs from owner-approved qualification')
    linux = value.get('Linux_3_12_3', {})
    if any(linux.get(key) != 'PASS' for key in ('wheel','sdist','LIVE_three_tasks','integrity','normal_fresh_pip_install')):
        raise ValueError('Missing required Linux/Python3.12 qualification')
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text('utf8')).get('project', {})
    if project.get('version') != VERSION or project.get('requires-python') != REQUIRED_PYTHON_SPEC:
        raise ValueError('Release metadata must match version2.0.0 and qualified Python3.12-3.14')
    return value


def preflight(require_ci=False):
    qualification_gate()
    from importlib import metadata
    sha = os.environ.get('GITHUB_SHA', '')
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise ValueError('Expected exact GitHub commit SHA')
    if git(['rev-parse', 'HEAD']) != sha:
        raise ValueError('Workflow checkout not at approved SHA')
    if git(['rev-parse', f'refs/tags/{TAG}^{{commit}}']) != sha:
        raise ValueError('Tag does not point to the workflow commit')
    if not (ROOT / 'pyproject.toml').read_text('utf8').count('version="2.0.0"') == 1:
        raise ValueError('Release metadata differs from 2.0.0')
    if os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch':
        raise ValueError('Manual workflow required')
    if os.environ.get('GITHUB_REF_NAME') not in ('main', TAG):
        raise ValueError('Unexpected ref; select main or the v2.0.0 tag')
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
    check_artifact_metadata(dist)
    return {name: {'sha256': check_hash(dist / name), 'bytes': (dist / name).stat().st_size}
            for name in (WHEEL, SDIST)}


def check_artifact_metadata(dist):
    """Read metadata without extracting archives, importing or fitting a model."""
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text('utf8'))['project']
    expected_dependencies = sorted(project.get('dependencies', []))
    limit = 256 * 1024
    wheel_metadata = 'morphtree-2.0.0.dist-info/METADATA'
    source_metadata = 'morphtree-2.0.0/PKG-INFO'
    native_suffixes = ('.c','.cc','.cpp','.cxx','.h','.hpp','.rs','.pyx','.pxd','.pyd','.so','.dll','.dylib','.a','.o','.obj','.lib','.exe','.wasm')
    try:
        if any((dist / name).stat().st_size > 32 * 1024**2 for name in (WHEEL, SDIST)):
            raise ValueError('archive size limit')
        with zipfile.ZipFile(dist / WHEEL) as archive:
            entries = archive.infolist()
            if len(entries) > 200 or len({x.filename for x in entries}) != len(entries):
                raise ValueError('duplicate or excessive wheel members')
            if any(x.filename.startswith('morphtree/') and x.filename.lower().endswith(native_suffixes) for x in entries):
                raise ValueError('non-Python proprietary wheel member')
            wheel_info=archive.getinfo('morphtree-2.0.0.dist-info/WHEEL')
            if wheel_info.file_size > limit:
                raise ValueError('oversized WHEEL metadata')
            wheel_headers=BytesParser(policy=policy.default).parsebytes(archive.read(wheel_info),headersonly=True)
            if wheel_headers.get_all('Root-Is-Purelib',[])!=['true'] or wheel_headers.get_all('Tag',[])!=['py3-none-any']:
                raise ValueError('wheel must be Pure Python py3-none-any')
            entry = archive.getinfo(wheel_metadata)
            if entry.file_size > limit:
                raise ValueError('oversized wheel metadata')
            wheel_raw = archive.read(entry)
        source_raw = None
        total = 0
        with tarfile.open(dist / SDIST, 'r|gz') as archive:
            for count, entry in enumerate(archive, 1):
                total += entry.size
                if count > 500 or entry.size < 0 or entry.size > 16 * 1024**2 or total > 32 * 1024**2:
                    raise ValueError('source inventory limit')
                if entry.name.startswith('morphtree-2.0.0/morphtree/') and entry.name.lower().endswith(native_suffixes):
                    raise ValueError('non-Python proprietary sdist member')
                if entry.name == source_metadata:
                    if source_raw is not None or not entry.isfile() or entry.size > limit:
                        raise ValueError('invalid or duplicate source metadata')
                    with archive.extractfile(entry) as stream:
                        source_raw = stream.read(limit + 1)
        if source_raw is None:
            raise ValueError('missing source metadata')
        normalize_spec = lambda value: sorted(x.strip() for x in value.split(','))
        for raw in (wheel_raw, source_raw):
            message = BytesParser(policy=policy.default).parsebytes(raw, headersonly=True)
            for key, expected in (('Name', 'morphtree'), ('Version', VERSION)):
                if message.get_all(key, []) != [expected]:
                    raise ValueError('incorrect or duplicate ' + key)
            specs = message.get_all('Requires-Python', [])
            if len(specs) != 1 or normalize_spec(specs[0]) != normalize_spec(REQUIRED_PYTHON_SPEC):
                raise ValueError('unqualified Requires-Python')
            if sorted(message.get_all('Requires-Dist', [])) != expected_dependencies:
                raise ValueError('dependencies differ from qualified source')
            if message.defects:
                raise ValueError('malformed metadata')
    except (ValueError, KeyError, OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise ValueError('Invalid release artifact metadata: ' + str(error)) from error


def make_manifest(base):
    qualification_gate()
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
    qualification_gate()
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
    sub.add_parser('check-qualification')
    q = sub.add_parser('preflight'); q.add_argument('--require-ci', action='store_true')
    q = sub.add_parser('make-manifest'); q.add_argument('folder')
    q = sub.add_parser('verify'); q.add_argument('folder'); q.add_argument('--origin-run-id'); q.add_argument('--require-origin-success', action='store_true')
    args = p.parse_args()
    if args.cmd == 'check-qualification':
        qualification_gate()
        print('PASS_RELEASE_QUALIFICATION')
    elif args.cmd == 'preflight':
        preflight(args.require_ci)
    elif args.cmd == 'make-manifest':
        make_manifest(args.folder)
    else:
        verify(args.folder, args.origin_run_id, args.require_origin_success)

if __name__ == '__main__':
    main()
