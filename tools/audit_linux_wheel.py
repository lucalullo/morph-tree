"""Audit Linux platform tags; reject any code/library change during repair.

Development tools only: auditwheel,pyelftools,packaging,patchelf. No downloads,
runtime dependency,training or publication. Failed audit work is retained.
"""
import argparse,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath

def file_map(path):
    values={}
    with zipfile.ZipFile(path) as z:
        for item in z.infolist():
            if PurePosixPath(item.filename).is_absolute() or '..' in PurePosixPath(item.filename).parts or '\\' in item.filename:raise ValueError('Unsafe ZIP member')
            if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('No ZIP symbolic links')
            if item.is_dir():
                if item.file_size:raise ValueError('Nonempty ZIP directory')
                continue
            if item.filename in values:raise ValueError('Duplicate ZIP file')
            values[item.filename]=z.read(item)
    return values

def validate_payload(before,after):
    changed=[n for n in set(before)|set(after) if before.get(n)!=after.get(n)]
    if any(not n.endswith(('.dist-info/WHEEL','.dist-info/RECORD')) for n in changed):raise ValueError('Audit changed code,library or other metadata')
    return sorted(changed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--wheel',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    if not sys.platform.startswith('linux'):raise RuntimeError('Linux audit requires an actual Linux environment')
    wheel=a.wheel.resolve();out=a.outdir.resolve();out.mkdir(parents=True,exist_ok=True);before=file_map(wheel)
    if not any(n.startswith('morphtree/') and n.endswith('.so') for n in before):raise ValueError('Expected MorphTree native ELF library')
    work=Path(tempfile.mkdtemp(prefix='morphtree-abi-audit-'));repaired=work/'repaired';repaired.mkdir()
    def command(name,args):
        with (work/(name+'.log')).open('x',encoding='utf8') as stream:r=subprocess.run([sys.executable,'-I','-B','-m','auditwheel',*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,timeout=45)
        if r.returncode:raise RuntimeError('auditwheel '+name+' failed; install the documented development tools;logs:'+str(work))
    command('show',['show',wheel]);command('repair',['repair','--plat','manylinux_2_17_x86_64','--wheel-dir',repaired,wheel])
    outputs=list(repaired.glob('*.whl'))
    if len(outputs)!=1:raise ValueError('One audited wheel required')
    result=outputs[0];changed=validate_payload(before,file_map(result));command('show-result',['show',result]);target=out/result.name
    if target.exists():raise FileExistsError('Preserve existing artifact: '+str(target))
    shutil.copyfile(result,target)
    print(json.dumps(dict(status='PASS_AUDITED_UNCHANGED_PAYLOAD',wheel=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed_files=changed,logs=str(work),model_fits=0,scope='ELF ABI/symbol/ISA policy;not whole-pipeline qualification on every Linux distribution')))

if __name__=='__main__':main()
