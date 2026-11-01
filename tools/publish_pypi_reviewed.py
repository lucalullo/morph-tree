"""Verification only for publish-pypi.yml. This module never uploads packages."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, urllib.error, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_publication_checks',ROOT/'tools/publish_reviewed.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
VERSION=shared.VERSION
PYPI='https://upload.pypi.org/legacy/'
PYPI_JSON='https://pypi.org/pypi/morphtree/2.1.0/json'
TEST_JSON='https://test.pypi.org/pypi/morphtree/2.1.0/json'
CONFIRMATION='PUBLISH morphtree 2.1.0 TO PYPI'

def get_json(url,token=None):
    if url!=PYPI_JSON:return shared.get_json(url,token)
    if token:shared.fail('Do not forward credentials to an index')
    request=urllib.request.Request(url,headers={'Accept':'application/json','User-Agent':'MorphTree-release-verifier/2.1.0'})
    with urllib.request.build_opener(shared.NoRedirect).open(request,timeout=20) as response:
        raw=response.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:shared.fail('Oversized index response')
        return json.loads(raw)

def require_indices(fetch=get_json):
    release=fetch(TEST_JSON)
    if release.get('info',{}).get('name','').lower()!='morphtree' or release['info'].get('version')!=VERSION or not release.get('urls'):
        shared.fail('Published TestPyPI 2.1.0 evidence required')
    if not any(not item.get('yanked',False) and shared.re.fullmatch('[0-9a-f]{64}',item.get('digests',{}).get('sha256','')) for item in release['urls']):
        shared.fail('Non-yanked TestPyPI distribution required')
    try:fetch(PYPI_JSON)
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
    else:shared.fail('Version already exists on official PyPI; stop without overwrite or retry')
    return dict(testpypi_release_verified=True,version=VERSION,pypi_HTTP_status=404,
        scope='TestPyPI release existence only; owner must review the new final-commit CI artifacts')

def require_policy(expected_version,root=ROOT):
    # Keep the legacy TestPyPI entry point and global preparation flags fail-closed.
    shared.require_policy('testpypi',expected_version,root)
    status=shared.read(root/'docs/RELEASE_STATUS.json')
    policy=status.get('pypi_workflow',{})
    if not(policy.get('prepared') is True and policy.get('environment')=='pypi'
        and policy.get('runtime_dispatch_confirmation')==CONFIRMATION
        and policy.get('publication_approved_by_preparation') is False):shared.fail('Dedicated PyPI preparation policy required')
    reference=shared.read(root/'docs/PYPI_RUNTIME_REFERENCE.json')
    expected=reference['files']
    actual={p.relative_to(root).as_posix():shared.sha(p) for p in (root/'python').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    for name in reference['build_files']:actual[name]=shared.sha(root/name)
    if actual!=expected:shared.fail('Qualified runtime, README or build configuration changed; separate review required')

def require_environment(environment):
    rules=[rule for rule in environment.get('protection_rules',[]) if rule.get('type')=='required_reviewers']
    if not any(item.get('type')=='User' and item.get('reviewer',{}).get('login')=='lucalullo' for rule in rules for item in rule.get('reviewers',[])):
        shared.fail('Required reviewer lucalullo missing from pypi environment')
    policy=environment.get('deployment_branch_policy') or {}
    if policy.get('protected_branches') is not False or policy.get('custom_branch_policies') is not True:
        shared.fail('pypi must restrict deployments to selected branch main')

def inspect_origin(commit,run_id,attempt,token,fetch=get_json):
    base=f'https://api.github.com/repos/{shared.REPOSITORY}/environments/pypi'
    def checked(url,credential):
        value=fetch(url,credential)
        if url==base:require_environment(value)
        return value
    value=shared.inspect_origin(commit,run_id,attempt,token,checked,environment_name='pypi')
    value.update(required_reviewer='lucalullo',deployment_branch='main',pypi_policy_verified=True,
        branch_verification='Workflow and CI enforce main; owner manages selected environment branch rules')
    return value

def require_context(commit,env=os.environ):
    if not(env.get('GITHUB_REPOSITORY')==shared.REPOSITORY and env.get('GITHUB_EVENT_NAME')=='workflow_dispatch'
        and env.get('GITHUB_REF')=='refs/heads/main' and env.get('GITHUB_SHA')==commit
        and env.get('GITHUB_WORKFLOW_REF')==shared.REPOSITORY+'/.github/workflows/publish-pypi.yml@refs/heads/main'):
        shared.fail('Exact reviewed main commit and dedicated publish-pypi.yml required')

def prepare_bundle(windows,linux,wsha,lsha,origin,out,root=ROOT,fetch=get_json):
    if not(origin.get('protected_environment')=='pypi' and origin.get('required_reviewers_verified')
        and origin.get('pypi_policy_verified') is True and origin.get('required_reviewer')=='lucalullo' and origin.get('deployment_branch')=='main'):
        shared.fail('Protected PyPI origin receipt required')
    evidence=require_indices(fetch)
    value=shared.prepare_bundle(windows,linux,wsha,lsha,origin,out,root,target='pypi')
    value.update(evidence,publication_workflow='.github/workflows/publish-pypi.yml',runtime_reference_sha256=shared.sha(root/'docs/PYPI_RUNTIME_REFERENCE.json'))
    (Path(out)/'APPROVED.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf8')
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['origin','preflight'],required=True)
    p.add_argument('--expected-version',required=True);p.add_argument('--reviewed-commit',required=True)
    p.add_argument('--build-run',required=True);p.add_argument('--build-attempt',type=int,required=True)
    p.add_argument('--origin',type=Path);p.add_argument('--windows',type=Path);p.add_argument('--linux',type=Path)
    p.add_argument('--windows-manifest-sha256');p.add_argument('--linux-manifest-sha256');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require_policy(a.expected_version);require_context(a.reviewed_commit)
    if a.stage=='origin':
        value=inspect_origin(a.reviewed_commit,a.build_run,a.build_attempt,os.environ['GH_TOKEN']);shared.publish(a.output,value)
    else:
        origin=shared.read(a.origin)
        if (origin['commit'],str(origin['build_run_id']),origin['build_run_attempt'])!=(a.reviewed_commit,a.build_run,a.build_attempt):shared.fail('Origin receipt mismatch')
        value=prepare_bundle(a.windows,a.linux,a.windows_manifest_sha256,a.linux_manifest_sha256,origin,a.output)
        subprocess.run([sys.executable,'-I','-B','-m','twine','check','--strict',*map(str,sorted((a.output/'packages').iterdir()))],check=True,timeout=90)
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a',encoding='utf8') as stream:stream.write('bundle_manifest_sha256='+shared.sha(a.output/'APPROVED.json')+'\n')
    print(json.dumps(value,sort_keys=True))

if __name__=='__main__':main()
