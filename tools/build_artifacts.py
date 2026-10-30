"""Build from a verified isolated copy; never mutate reviewed sources."""
import argparse, importlib.util, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,default=ROOT/'dist')
    parser.add_argument('--manylinux',action='store_true',help='Linux-only auditwheel repair, with unchanged package/library payload')
    args=parser.parse_args();out=args.outdir.resolve()
    if args.manylinux and not sys.platform.startswith('linux'):raise RuntimeError('--manylinux requires an actual Linux build environment')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be empty; preserve existing artifacts')
    spec=importlib.util.spec_from_file_location('source_inventory',ROOT/'tools/check_repository.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);files=module.check()
    work=Path(tempfile.mkdtemp(prefix='morphtree-source-build-'));source=work/'source';source.mkdir()
    for name in [*files,'SOURCE_MANIFEST.json']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',VSCMD_SKIP_SENDTELEMETRY='1')
    # Source archive first: contains the reviewed source inventory, not local build state.
    for kind in ('sdist','wheel'):
        destination=work/'raw-wheel' if kind=='wheel' and args.manylinux else out
        with (work/(kind+'.log')).open('x',encoding='utf8') as log:
            result=subprocess.run([sys.executable,'-I','-B','-m','build','--no-isolation',
                                   '--'+kind,'--outdir',str(destination)],cwd=source,env=env,
                                  stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode:raise RuntimeError('Build failed once; logs retained at '+str(work))
    if args.manylinux:
        wheels=list((work/'raw-wheel').glob('*.whl'))
        if len(wheels)!=1:raise RuntimeError('Exactly one raw Linux wheel required')
        with (work/'audit.log').open('x',encoding='utf8') as log:
            result=subprocess.run([sys.executable,'-I','-B',str(source/'tools/audit_linux_wheel.py'),'--wheel',str(wheels[0]),'--outdir',str(out)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150)
        if result.returncode:raise RuntimeError('Linux ABI audit failed;raw wheel/logs retained at '+str(work))
    module.check()
    print(json.dumps({'status':'BUILT_NOT_YET_INSTALLED_TESTED','artifacts':str(out),
                      'retained_build_logs':str(work),'model_fits':0}))
if __name__=='__main__':main()
