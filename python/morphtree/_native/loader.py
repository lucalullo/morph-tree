"""Load only the installed package's hash-bound library;never compile at import."""
import ctypes,hashlib,json,platform,struct,sys,threading
from pathlib import Path
from .runtime import Runtime
_LOCK=threading.RLock();_RUNTIME=None;_INFO=None
def platform_key():
    if struct.calcsize('P')!=8 or platform.machine().lower() not in ('amd64','x86_64'):raise ImportError('This native build requires64bit x86_64')
    if sys.platform=='win32':return 'win_amd64'
    if sys.platform.startswith('linux'):return 'linux_x86_64'
    raise ImportError('No qualified native artifact for this operating system')
def artifact():
    root=Path(__file__).resolve().parent;manifest=root/'manifest.json'
    if not manifest.is_file():raise ImportError('Native library missing:install the matching experimental wheel or build from source with an existing C toolchain')
    v=json.loads(manifest.read_text('utf8'));expected={'schema','ABI','platform','library','binary_sha256','sources','floating_point','runtime_compilation','build_settings'}
    if type(v) is not dict or set(v)!=expected or v['schema']!='MORPHTREE_NATIVE_ABI_1' or v['ABI']!=1 or v['platform']!=platform_key() or v['runtime_compilation'] is not False:raise ImportError('Native artifact capability mismatch')
    name='morphtree_kernels.dll' if sys.platform=='win32' else 'libmorphtree_kernels.so'
    if v['library']!='bin/'+v['platform']+'/'+name or set(v['sources'])!={'src/histogram.c','src/traversal.c'}:raise ImportError('Unexpected native artifact path/source contract')
    p=root/v['library']
    for item in (p,*p.parents):
        if not item.is_relative_to(root):break
        if item.is_symlink() or getattr(item,'is_junction',lambda:False)():raise ImportError('Native paths cannot be links')
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=v['binary_sha256']:raise ImportError('Native binary integrity failure')
    for n,h in v['sources'].items():
        source=root/n
        if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=h:raise ImportError('Native source integrity failure')
    return p,v
def get_runtime():
    global _RUNTIME,_INFO
    with _LOCK:
        if _RUNTIME is None:
            p,v=artifact()
            # Retain GIL;Windows lookup uses the verified DLL directory and
            # standard loader directories,never the current working directory.
            library=ctypes.PyDLL(str(p),winmode=0x1100) if sys.platform=='win32' else ctypes.PyDLL(str(p),mode=ctypes.RTLD_LOCAL)
            _RUNTIME=Runtime(library);_INFO=v
        return _RUNTIME
def runtime_info():
    get_runtime()
    return dict(platform=_INFO['platform'],binary_sha256=_INFO['binary_sha256'],ABI=_INFO['ABI'],runtime_compilation=False,cache=_RUNTIME.trees.report())
