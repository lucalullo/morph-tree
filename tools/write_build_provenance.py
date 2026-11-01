"""CI-only artifact manifest; emit after build, installed tests and Twine pass."""
import argparse, hashlib, json, os, re, sys, tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(env,files,root=ROOT,platform=None,python=None):
    if not re.fullmatch('[0-9a-f]{40}',env['GITHUB_SHA']) or env['GITHUB_EVENT_NAME']!='workflow_dispatch':raise ValueError('Explicit CI commit required')
    if env['GITHUB_REF']!='refs/heads/main':raise ValueError('Main branch build required')
    if env['GITHUB_WORKFLOW_REF']!=env['GITHUB_REPOSITORY']+'/.github/workflows/test.yml@refs/heads/main':raise ValueError('Qualified build workflow required')
    if len(files)!=2 or sum(p.name.endswith('.whl') for p in files)!=1 or sum(p.name.endswith('.tar.gz') for p in files)!=1:raise ValueError('One wheel and one sdist required')
    return dict(schema='MORPHTREE_BUILD_PROVENANCE_1',repository=env['GITHUB_REPOSITORY'],commit=env['GITHUB_SHA'],
        build_run_id=int(env['GITHUB_RUN_ID']),build_run_attempt=int(env['GITHUB_RUN_ATTEMPT']),workflow='.github/workflows/test.yml',
        event=env['GITHUB_EVENT_NAME'],ref=env['GITHUB_REF'],platform=platform or env['RUNNER_OS'],python=python or '.'.join(map(str,sys.version_info[:2])),
        version=tomllib.loads((root/'pyproject.toml').read_text('utf8'))['project']['version'],source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),
        README_sha256=sha(root/'README.md'),validation='ALL_BUILD_AND_INSTALLED_CHECKS_PASSED',
        files={p.name:dict(sha256=sha(p),size=p.stat().st_size) for p in sorted(files)})
def main():
    p=argparse.ArgumentParser();p.add_argument('--dist',type=Path,default=ROOT/'dist');a=p.parse_args()
    value=manifest(os.environ,sorted(a.dist.glob('*.whl'))+sorted(a.dist.glob('*.tar.gz')))
    with (a.dist/'BUILD_PROVENANCE.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
