"""MorphTree: own native strategies and a bounded evolving learning program."""
from .config import AlgorithmGenome,HyperParams,MorphTreeConfig,VERSION as __native_core_version__
from .model import MorphTreeClassifier as NativeMorphTreeClassifier,MorphTreeRegressor as NativeMorphTreeRegressor,MorphTreeMulticlassClassifier,AutoMorphTree
from .persistence import LoadLimits
from ._installed import activate as _activate_installed
_activate_installed(__file__)
from .api import MorphTree,TrainingNotEnabledError
from .estimators import MorphTreeClassifier,MorphTreeRegressor
__version__='2.0.0'
__all__=['MorphTreeClassifier','MorphTreeRegressor','MorphTree','load_model','AlgorithmGenome','HyperParams','MorphTreeConfig','LoadLimits','__version__']
__runtime__={'release_candidate':False,'backend':'pure','scientific_training_authority':0}
def load_model(path,*,sha256=None):
    """Load the public model with bounded JSON validation; a digest is optional."""
    return MorphTree.load(path,sha256=sha256)
