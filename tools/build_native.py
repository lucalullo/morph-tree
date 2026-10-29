"""Local native build/identity helper. Never installs or downloads a compiler."""
import ast,hashlib,json,os,platform,shutil,subprocess,sys,sysconfig,time,secrets,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'python/morphtree';NATIVE=PACKAGE/'_native'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf8')
def platform_key():
    if sys.maxsize<=2**32 or platform.machine().lower() not in ('amd64','x86_64'):raise RuntimeError('Only64bit x86_64 native builds are currently supported')
    if sys.platform=='win32':return 'win_amd64'
    if sys.platform.startswith('linux'):return 'linux_x86_64'
    raise RuntimeError('No authentic native build qualification for this platform')

def msvc_setup():
    # Bounded official discovery,no account/configuration access or installation.
    base=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))
    finder=base/'Microsoft Visual Studio/Installer/vswhere.exe'
    if finder.is_file():
        result=subprocess.run([str(finder),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],capture_output=True,text=True,timeout=15,check=True)
        for value in result.stdout.splitlines():
            p=Path(value)/'VC/Auxiliary/Build/vcvars64.bat'
            if p.is_file():return p
    # The owner's verified location is also usable in an isolated source build.
    p=base/'Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    if p.is_file():return p
    raise RuntimeError('MSVC x64+Windows SDK required only for source builds;install a compatible prebuilt wheel to avoid a compiler')

def build():
    key=platform_key();sources=[NATIVE/'src/histogram.c',NATIVE/'src/traversal.c']
    before={p.name:sha(p) for p in sources};work=ROOT/'.native-builds'/('build-'+secrets.token_hex(6));work.mkdir(parents=True);start=time.monotonic()
    filename='morphtree_kernels.dll' if key=='win_amd64' else 'libmorphtree_kernels.so';target=work/filename
    if key=='win_amd64':
        setup=msvc_setup();flags=['/nologo','/LD','/std:c11','/W4','/WX','/O2','/fp:strict','/MT','/Brepro']
        command=work/'build.cmd';command.write_text('\r\n'.join(['@echo off',f'call "{setup}" > "{work/"setup.log"}" 2>&1','if errorlevel 1 exit /b 1',f'where cl > "{work/"compiler-path.txt"}"','if errorlevel 1 exit /b 1',f'cl {" ".join(flags)} '+ ' '.join('"'+str(p)+'"' for p in sources)+f' /Fe:"{target}" /link /INCREMENTAL:NO /Brepro > "{work/"compiler.log"}" 2>&1','exit /b %errorlevel%'])+'\r\n',encoding='utf8')
        environment={k:v for k,v in os.environ.items() if k.upper() not in {'CL','_CL_','LINK','_LINK_','CFLAGS','CXXFLAGS'}}
        environment['VSCMD_SKIP_SENDTELEMETRY']='1'
        result=subprocess.run([os.environ.get('COMSPEC','C:/Windows/System32/cmd.exe'),'/d','/c',str(command)],cwd=work,env=environment,capture_output=True,text=True,timeout=90)
        compiler=next((v.strip() for v in (work/'compiler-path.txt').read_text('utf8').splitlines() if v.strip()),None) if (work/'compiler-path.txt').exists() else None
        settings=dict(flags=flags,setup_sha256=sha(setup),compiler_sha256=sha(Path(compiler)) if compiler else None,CRT='static /MT')
    else:
        compiler=next((shutil.which(v) for v in ('cc','gcc','clang') if shutil.which(v)),None)
        if not compiler:raise RuntimeError('No installed Linux C compiler;no installation or automatic retry attempted')
        flags=['-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-fPIC','-fvisibility=hidden','-Wall','-Wextra','-Werror','-shared']
        result=subprocess.run([compiler,*flags,*(str(p) for p in sources),'-o',str(target)],cwd=work,capture_output=True,text=True,timeout=90);settings=dict(flags=flags,compiler_sha256=sha(Path(compiler)))
    (work/'launcher.log').write_text(result.stdout+result.stderr,encoding='utf8')
    receipt=dict(status='BUILT_NOT_YET_QUALIFIED' if result.returncode==0 else 'FAILED_NO_RETRY',platform=key,sources=before,settings=settings,seconds=time.monotonic()-start,returncode=result.returncode,system_installations=0,network_operations=0,model_fits=0)
    write(work/'BUILD.json',receipt)
    if result.returncode:raise RuntimeError('Native build failed once;preserved logs:'+str(work))
    if {p.name:sha(p) for p in sources}!=before:raise RuntimeError('C sources changed during build')
    dest=NATIVE/'bin'/key/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,dest)
    manifest=dict(schema='MORPHTREE_NATIVE_ABI_1',ABI=1,platform=key,library=dest.relative_to(NATIVE).as_posix(),binary_sha256=sha(dest),sources={'src/'+k:h for k,h in before.items()},floating_point='binary64 ordered / strict,without reassociation or contracted multiply-add',runtime_compilation=False,build_settings=settings)
    write(NATIVE/'manifest.json',manifest)
    return dict(**receipt,binary_sha256=sha(dest),library=str(dest),build_receipt=str(work/'BUILD.json'))

def refresh_identity():
    version=tomllib.loads((ROOT/'pyproject.toml').read_text('utf8'))['project']['version']
    for filename,name in [('__init__.py','__version__'),('api.py','PUBLIC_USER_RUNTIME')]:
        values=[ast.literal_eval(n.value) for n in ast.parse((PACKAGE/filename).read_text('utf8')).body
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        if values!=[version]:raise RuntimeError('Package/API version differs from build metadata: '+filename)
    manifest=PACKAGE/'_experimental/manifest.json';value=json.loads(manifest.read_text('utf8'))
    value['files']={p.relative_to(PACKAGE).as_posix():sha(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p!=manifest and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    # Qualification is reviewed evidence, never inferred or reset by compilation.
    value.update(version=version,backend='automatic-native');write(manifest,value)
    source=ROOT/'python/SOURCE_MANIFEST.json';v=json.loads(source.read_text('utf8'))
    v.update(version=version,release_ready=False,base_qualified_version='2.1.0')
    v['files']={p.relative_to(ROOT/'python').as_posix():sha(p) for p in sorted((ROOT/'python').rglob('*')) if p.is_file() and p!=source and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')};write(source,v)
    return sha(manifest)

if __name__=='__main__':
    result=build();result['engine_manifest_sha256']=refresh_identity();print(json.dumps(result))
