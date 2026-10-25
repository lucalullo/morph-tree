import hashlib,json
from pathlib import Path
def activate(file):
    package=Path(file).resolve().parent
    m=json.loads((package/'_experimental/manifest.json').read_text('utf8'))
    actual={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=set(m['files'])|{'_experimental/manifest.json'}:raise ImportError('Missing or unexpected installed file')
    for n,h in m['files'].items():
        if hashlib.sha256((package/n).read_bytes()).hexdigest()!=h:raise ImportError('Modified installed component: '+n)
    from .model import MorphTreeClassifier,MorphTreeRegressor,MorphTreeMulticlassClassifier,AutoMorphTree
    from ._experimental.admission import AdmissionError
    def denied(*a,**k):raise AdmissionError({'status':'BLOCKED','reason':'INSTALLED_NATIVE_TRAINING_UNQUALIFIED_ZERO_BUDGET'})
    from ._experimental.execution_v2 import native_fit_entry
    for cls in (MorphTreeClassifier,MorphTreeRegressor,MorphTreeMulticlassClassifier):cls.fit=native_fit_entry(cls.fit,cls)
    AutoMorphTree.fit=denied
    from ._experimental.facade import MorphTree
    return MorphTree,{'scope':'RESEARCH_ONLY_ZERO_FIT','portable_inference':True,'physical_training':False}
