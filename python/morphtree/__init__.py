"""MorphTree: own native strategies and a bounded evolving learning program."""
from .config import AlgorithmGenome,HyperParams,MorphTreeConfig,VERSION as __native_core_version__
from .model import MorphTreeClassifier as NativeMorphTreeClassifier,MorphTreeRegressor as NativeMorphTreeRegressor,MorphTreeMulticlassClassifier,AutoMorphTree
from .persistence import LoadLimits
from ._installed import activate as _activate_installed
_activate_installed(__file__)
from .api import MorphTree,TrainingNotEnabledError
from .estimators import MorphTreeClassifier,MorphTreeRegressor
__version__='2.1.0'
__all__=['MorphTreeClassifier','MorphTreeRegressor','MorphTree','load_model','AlgorithmGenome','HyperParams','MorphTreeConfig','LoadLimits','__version__']
__runtime__={'release_candidate':True,'backend':'automatic-native',
             'implementation':'Python with verified C kernels',
             'backend_policy':'C histograms and cached traversal; Python traversal below 128 rows',
             'native_loading':'Lazy; runtime_info() reports the loaded native library',
             'qualification':'Documented OS, interpreter, task and scale scope only',
             'reference_version':'2.1.0 (historical Python implementation)',
             'publication_approved':False,'scientific_training_authority':0}
def load_model(path,*,sha256=None):
    """Load the public model with bounded JSON validation; a digest is optional."""
    return MorphTree.load(path,sha256=sha256)

# Scalable public task facades; retain the legacy MorphTree/artifact decoder.
from .scalable import MorphTreeClassifier,MorphTreeRegressor
__runtime__.update(private_research=False,scope="ORDINARY_USER_WITHOUT_SCIENTIFIC_AUTHORITY")
def load_model(path,*,sha256=None):
    """Load current and historical models with validated bounded JSON."""
    from .scalable import _Scalable
    return _Scalable.load(path,sha256=sha256)
