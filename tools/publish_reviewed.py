"""Owner-triggered publication gate. This tool is not run during local preparation."""
import argparse,hashlib,json,re,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['testpypi','pypi'],required=True)
    p.add_argument('--expected-version',required=True);a=p.parse_args()
    status=json.loads((ROOT/'docs/RELEASE_STATUS.json').read_text('utf8'))
    version=tomllib.loads((ROOT/'pyproject.toml').read_text('utf8'))['project']['version']
    if not (status['status']=='GO' and status['version_approved'] and status['publication_approved']
            and status['version']==version==a.expected_version and re.fullmatch(r'\d+\.\d+\.\d+',version)):
        raise SystemExit('Publication blocked: approved final version and GO required')
    if a.target=='pypi' and status.get('testpypi_verified_version')!=version:
        raise SystemExit('Publication blocked: TestPyPI verification required')
    candidates=sorted((ROOT/'dist').rglob('*.whl'))+sorted((ROOT/'dist').rglob('*.tar.gz'))
    expected=status.get('reviewed_artifact_sha256',{})
    unique={}
    for artifact in candidates:
        if artifact.name not in expected or hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected[artifact.name]:
            raise SystemExit('Unreviewed artifact or checksum mismatch')
        unique[artifact.name]=artifact
    if not expected or set(unique)!=set(expected):raise SystemExit('Missing reviewed distributions')
    artifacts=list(unique.values())
    subprocess.run([sys.executable,'-m','twine','check',*map(str,artifacts)],check=True)
    subprocess.run([sys.executable,'-m','twine','upload','--non-interactive','--repository',a.target,*map(str,artifacts)],check=True)
if __name__=='__main__':main()
