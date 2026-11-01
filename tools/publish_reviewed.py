"""Verification only: upload belongs exclusively to the protected OIDC job."""
import argparse, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tarfile, tomllib
import urllib.error, urllib.request, zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='lucalullo/morph-tree'
VERSION='2.1.0'
TESTPYPI='https://test.pypi.org/legacy/'
CONFIRMATION='PUBLISH morphtree 2.1.0 TO TESTPYPI'
PROFILES={f'zero-fit-{s}-{v}' for s in ('windows-latest','ubuntu-24.04') for v in ('3.12','3.13','3.14')}
ARTIFACTS={f'morphtree-{s}-{v}' for s in ('windows-latest','ubuntu-24.04') for v in ('3.12','3.13','3.14')}

def fail(message): raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text('utf8'))
def publish(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def version_allowed(version,target):return target=='testpypi' and version==VERSION

def require_policy(target,expected_version,root=ROOT):
    if target!='testpypi':fail('Official PyPI is hard-blocked; TestPyPI only')
    project=tomllib.loads((root/'pyproject.toml').read_text('utf8'))['project'];status=read(root/'docs/RELEASE_STATUS.json')
    if not(expected_version==project['version']==status['version']==VERSION and status['version_approved']
        and status['current_distribution_training_verified'] and status['local_technical_status']=='GO_SCOPED_LOCAL_DELIVERY'):
        fail('Version or qualification mismatch')
    if not status.get('testpypi_workflow',{}).get('prepared'):fail('TestPyPI workflow not prepared')
    if status.get('pypi_upload_enabled') is not False or status['publication_approved'] is not False:
        fail('Official PyPI must remain blocked')
    if project['readme']!={'file':'README.md','content-type':'text/markdown'}:fail('Approved README required')
    return project

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):fail('Unexpected redirect; stop without retry')

def get_json(url,token=None):
    if not(url.startswith('https://api.github.com/repos/'+REPOSITORY+'/') or url==f'https://test.pypi.org/pypi/morphtree/{VERSION}/json'):
        fail('Verification URL outside allowed services')
    headers={'Accept':'application/json','User-Agent':'MorphTree-release-verifier/2.1.0'}
    if token:
        if not url.startswith('https://api.github.com/'):fail('Do not forward GitHub credentials')
        headers.update(Authorization='Bearer '+token,**{'X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url,headers=headers),timeout=20) as response:
        raw=response.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:fail('Oversized service response')
        return json.loads(raw)

def require_version_absent(fetch=get_json):
    try:fetch(f'https://test.pypi.org/pypi/morphtree/{VERSION}/json')
    except urllib.error.HTTPError as error:
        if error.code==404:return dict(index='testpypi',version=VERSION,HTTP_status=404,absent=True)
        raise
    fail('Version already exists on TestPyPI; no overwrite, skip or retry')

def validate_origin(run,jobs,artifacts,environment,commit,run_id,attempt):
    if not re.fullmatch('[0-9a-f]{40}',commit):fail('Full reviewed commit SHA required')
    if not(str(run['id'])==str(run_id) and run['run_attempt']==attempt
        and run['repository']['full_name']==run['head_repository']['full_name']==REPOSITORY
        and run['head_sha']==commit and run['head_branch']=='main'
        and run['path']=='.github/workflows/test.yml' and run['event']=='workflow_dispatch'
        and run['status']=='completed' and run['conclusion']=='success'):
        fail('CI source, attempt, workflow or conclusion mismatch')
    actual={j['name']:j for j in jobs}
    if len(actual)!=len(jobs) or not PROFILES<=actual.keys():fail('Six distinct CI profiles required')
    for name in PROFILES:
        j=actual[name]
        if j['status']!='completed' or j['conclusion']!='success' or j['head_sha']!=commit:fail('CI job failed/skipped or wrong commit')
    names=[a['name'] for a in artifacts]
    if set(names)!=ARTIFACTS or len(names)!=6:fail('Six immutable artifact sets without duplicates required')
    for a in artifacts:
        if a['expired'] or type(a['id']) is not int or a['id']<=0 or a['size_in_bytes']<=0:fail('Invalid/expired build artifact')
        if a.get('workflow_run',{}).get('head_sha',commit)!=commit:fail('Artifact commit mismatch')
    rules=[r for r in environment.get('protection_rules',[]) if r.get('type')=='required_reviewers' and r.get('reviewers')]
    if environment.get('name')!='testpypi' or not rules:fail('Protected testpypi environment with required reviewers required')
    return dict(schema='MORPHTREE_CI_ORIGIN_1',repository=REPOSITORY,commit=commit,build_run_id=int(run_id),
        build_run_attempt=attempt,workflow='.github/workflows/test.yml',required_jobs=sorted(PROFILES),
        artifacts=artifacts,protected_environment='testpypi',required_reviewers_verified=True)

def inspect_origin(commit,run_id,attempt,token,fetch=get_json):
    if not re.fullmatch('[1-9][0-9]*',str(run_id)) or attempt<1:fail('Positive run ID/attempt required')
    base=f'https://api.github.com/repos/{REPOSITORY}'
    run=fetch(f'{base}/actions/runs/{run_id}',token)
    jobs=fetch(f'{base}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100',token)
    artifacts=fetch(f'{base}/actions/runs/{run_id}/artifacts?per_page=100',token)
    if jobs['total_count']!=len(jobs['jobs']) or artifacts['total_count']!=len(artifacts['artifacts']):fail('Unexpected pagination')
    environment=fetch(f'{base}/environments/testpypi',token)
    return validate_origin(run,jobs['jobs'],artifacts['artifacts'],environment,commit,run_id,attempt)

def check_build_manifest(folder,approved_sha,origin,system,root=ROOT):
    folder=Path(folder);manifest=folder/'BUILD_PROVENANCE.json'
    if folder.is_symlink() or not folder.is_dir() or manifest.is_symlink() or not manifest.is_file():fail('Unsafe build directory or manifest link')
    if not re.fullmatch('[0-9a-f]{64}',approved_sha) or sha(manifest)!=approved_sha:fail('Unapproved build manifest SHA-256')
    value=read(manifest)
    required=dict(schema='MORPHTREE_BUILD_PROVENANCE_1',repository=REPOSITORY,commit=origin['commit'],
        build_run_id=origin['build_run_id'],build_run_attempt=origin['build_run_attempt'],workflow='.github/workflows/test.yml',
        event='workflow_dispatch',ref='refs/heads/main',platform=system,python='3.12',version=VERSION,
        source_manifest_sha256=sha(root/'SOURCE_MANIFEST.json'),README_sha256=sha(root/'README.md'),validation='ALL_BUILD_AND_INSTALLED_CHECKS_PASSED')
    if any(value.get(k)!=v for k,v in required.items()):fail('Build manifest source/provenance mismatch')
    files=value.get('files',{})
    if {p.name for p in folder.iterdir()}!={*files,'BUILD_PROVENANCE.json'} or len(files)!=2:fail('Extra/missing build files')
    for name,m in files.items():
        p=folder/name
        if Path(name).name!=name or '\\' in name or p.is_symlink() or not p.is_file():fail('Unsafe artifact filename/link')
        if sha(p)!=m['sha256'] or p.stat().st_size!=m['size']:fail('Artifact checksum/size mismatch')
    wheels=[folder/n for n in files if n.endswith('.whl')];sdists=[folder/n for n in files if n==f'morphtree-{VERSION}.tar.gz']
    if len(wheels)!=1 or len(sdists)!=1:fail('One wheel and one sdist per platform required')
    if system=='Windows' and not wheels[0].name.endswith('-win_amd64.whl'):fail('Wrong Windows platform')
    if system=='Linux' and ('manylinux' not in wheels[0].name or 'x86_64' not in wheels[0].name):fail('Audited x86_64 manylinux wheel required')
    return wheels[0],sdists[0]

def safe_wheel(path):
    with zipfile.ZipFile(path) as archive:
        seen=set()
        for member in archive.infolist():
            name=PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename or member.filename in seen or (member.external_attr>>16)&0o170000==0o120000:
                fail('Unsafe/duplicate wheel member')
            seen.add(member.filename)

def sdist_payload(path):
    result={}
    with tarfile.open(path) as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or member.name in result or not member.isfile():fail('Unsafe/duplicate source archive member')
            result[member.name]=archive.extractfile(member).read()
    return result

def prepare_bundle(windows,linux,windows_sha,linux_sha,origin,out,root=ROOT):
    spec=importlib.util.spec_from_file_location('distribution_review',root/'tools/check_distribution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ww,ws=check_build_manifest(windows,windows_sha,origin,'Windows',root)
    lw,ls=check_build_manifest(linux,linux_sha,origin,'Linux',root)
    for wheel in (ww,lw):safe_wheel(wheel)
    payload=sdist_payload(ws)
    if payload!=sdist_payload(ls):fail('Platform sdists differ in contents; no silent deduplication')
    prefix=f'morphtree-{VERSION}/';source=dict(read(root/'SOURCE_MANIFEST.json')['files']);source['SOURCE_MANIFEST.json']=sha(root/'SOURCE_MANIFEST.json')
    if set(payload)!={prefix+n for n in source}|{prefix+'PKG-INFO'}:fail('Sdist source membership mismatch')
    for name,pin in source.items():
        if hashlib.sha256(payload[prefix+name]).hexdigest()!=pin:fail('Sdist source bytes differ from this commit')
    for p in (ww,lw,ws,ls):module.check_artifact(p)
    selected=[ww,lw,ws]  # Canonical Windows sdist; Linux source independently verified.
    if len({p.name for p in selected})!=3:fail('Upload filenames must be unique')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);packages=out/'packages';packages.mkdir()
    for p in selected:shutil.copyfile(p,packages/p.name)
    value=dict(schema='MORPHTREE_TESTPYPI_APPROVAL_1',repository=REPOSITORY,commit=origin['commit'],version=VERSION,
        target='testpypi',repository_url=TESTPYPI,build_run_id=origin['build_run_id'],build_run_attempt=origin['build_run_attempt'],
        windows_manifest_sha256=windows_sha,linux_manifest_sha256=linux_sha,canonical_sdist='Windows; both platform payloads verified identical',
        files={p.name:sha(p) for p in selected})
    publish(out/'APPROVED.json',value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['testpypi','pypi'],default='testpypi')
    p.add_argument('--expected-version',default=VERSION);p.add_argument('--stage',choices=['origin','preflight']);p.add_argument('--check-testpypi',action='store_true')
    p.add_argument('--reviewed-commit');p.add_argument('--build-run');p.add_argument('--build-attempt',type=int,default=1)
    p.add_argument('--origin',type=Path);p.add_argument('--windows',type=Path);p.add_argument('--linux',type=Path)
    p.add_argument('--windows-manifest-sha256');p.add_argument('--linux-manifest-sha256');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.target!='testpypi':raise SystemExit('Publication blocked: official PyPI is hard-blocked')
    if a.check_testpypi:print(json.dumps(require_version_absent()));return
    if not a.stage:raise SystemExit('Publication blocked: verification-only tool; use protected TestPyPI workflow')
    require_policy(a.target,a.expected_version)
    if not(os.environ.get('GITHUB_REPOSITORY')==REPOSITORY and os.environ.get('GITHUB_EVENT_NAME')=='workflow_dispatch'
        and os.environ.get('GITHUB_REF')=='refs/heads/main' and os.environ.get('GITHUB_SHA')==a.reviewed_commit):fail('Same reviewed main-branch commit required')
    if a.stage=='origin':
        value=inspect_origin(a.reviewed_commit,a.build_run,a.build_attempt,os.environ['GH_TOKEN']);publish(a.output,value)
    else:
        origin=read(a.origin)
        if (origin['commit'],str(origin['build_run_id']),origin['build_run_attempt'])!=(a.reviewed_commit,a.build_run,a.build_attempt):fail('Origin receipt mismatch')
        require_version_absent()
        value=prepare_bundle(a.windows,a.linux,a.windows_manifest_sha256,a.linux_manifest_sha256,origin,a.output)
        subprocess.run([sys.executable,'-I','-B','-m','twine','check','--strict',*map(str,sorted((a.output/'packages').iterdir()))],check=True,timeout=90)
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a',encoding='utf8') as f:f.write('bundle_manifest_sha256='+sha(a.output/'APPROVED.json')+'\n')
    print(json.dumps(value,sort_keys=True))

if __name__=='__main__':main()
