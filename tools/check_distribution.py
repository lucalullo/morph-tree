"""Check that wheel/sdist descriptions equal the approved README.md."""
import argparse,email.policy,hashlib,json,tarfile,tomllib,zipfile
from email.parser import BytesParser
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def normalized(data):return data.replace(b'\r\n',b'\n')

def validate_metadata(raw,readme,version):
    m=BytesParser(policy=email.policy.compat32).parsebytes(raw)
    if m.get('Name')!='morphtree' or m.get('Version')!=version:raise ValueError('Distribution name/version mismatch')
    if m.get('Description-Content-Type','').split(';',1)[0].strip().lower()!='text/markdown':raise ValueError('Markdown description required')
    body=m.get_payload(decode=True)
    if body is None or normalized(body)!=normalized(readme):raise ValueError('Description differs from approved README.md')
    return dict(version=version,description_sha256=hashlib.sha256(normalized(body)).hexdigest(),content_type=m.get('Description-Content-Type'))

def check_artifact(path):
    project=tomllib.loads((ROOT/'pyproject.toml').read_text('utf8'))['project']
    if project['readme']!={'file':'README.md','content-type':'text/markdown'}:raise ValueError('README.md must be the sole description source')
    if (ROOT/'README_PYPI.md').exists():raise ValueError('Obsolete duplicate README_PYPI.md present')
    raw=(ROOT/'README.md').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if json.loads((ROOT/'SOURCE_MANIFEST.json').read_text('utf8'))['files']['README.md']!=digest:raise ValueError('README source hash mismatch')
    if path.name.endswith('.whl'):
        with zipfile.ZipFile(path) as z:
            entries=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(entries)!=1:raise ValueError('Exactly one wheel METADATA required')
            meta=z.read(entries[0]);contract=json.loads(z.read('morphtree/_experimental/manifest.json'))['public_description']
            if contract['README_sha256']!=digest:raise ValueError('Packaged README approval differs')
    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path,'r:gz') as t:
            prefix='morphtree-'+project['version']+'/'
            meta=t.extractfile(prefix+'PKG-INFO').read()
            if t.extractfile(prefix+'README.md').read()!=raw:raise ValueError('Sdist README bytes differ')
            if any(m.name.endswith('/README_PYPI.md') for m in t.getmembers()):raise ValueError('Duplicate README in sdist')
    else:raise ValueError('Expected wheel or sdist')
    result=validate_metadata(meta,raw,project['version'])
    result.update(status='PASS_APPROVED_README_DESCRIPTION',artifact=path.name,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),README_sha256=digest)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('artifacts',nargs='+',type=Path);a=p.parse_args();items=[]
    for item in a.artifacts:items.extend(sorted(item.glob('*.whl'))+sorted(item.glob('*.tar.gz')) if item.is_dir() else [item])
    if not items:raise ValueError('No artifacts supplied')
    for item in items:print(json.dumps(check_artifact(item),sort_keys=True))
if __name__=='__main__':main()
