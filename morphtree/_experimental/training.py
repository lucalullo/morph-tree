"""Existing native estimators/aggregation behind an UNADMITTED fit boundary.

No historical worker is relaunched. A future source-bound v2 worker must qualify
and supply execute_native before any physical training. No admission is issued
by constructing the estimator, by a deadline or by this module's existence.
"""
from dataclasses import replace, asdict
from morphtree import MorphTreeClassifier, MorphTreeMulticlassClassifier, MorphTreeRegressor
from morphtree.config import AlgorithmGenome, HyperParams, MorphTreeConfig
from .support import FamilyModel
from .sampling import bootstrap

NATIVE={'binary':MorphTreeClassifier,'multiclass':MorphTreeMulticlassClassifier,'regression':MorphTreeRegressor}


class TrainingNotAdmitted(RuntimeError):
    pass


class NativeExecutor:
    evidence_level='NATIVE_PATH_IMPLEMENTED_NOT_FIT_ADMITTED'

    def execute_native(self, estimator, X, y, *, identity, cancel):
        # This boundary must be replaced ONLY by a separately qualified finite
        # source/interpreter/protocol/TRAIN/budget-bound v2 execution adapter.
        raise TrainingNotAdmitted('MORPH-026 has zero training budget; separate owner smoke authorization and qualified v2 worker required')

    def fit_family(self, task, family, X, y, *, seed, params, check, identity, genome=None):
        if task not in NATIVE or family not in ('boost','bag') or not isinstance(params,HyperParams) or params.n_estimators>8:
            raise ValueError('Known native family/task and at most eight stages/members in this finite adapter')
        genome=AlgorithmGenome() if genome is None else genome
        from morphtree.config import validate_task_genome
        validate_task_genome(task,genome)
        check(); members=[]
        rounds=params.n_estimators
        for i in range(1 if family=='boost' else rounds):
            check()
            ids=list(range(len(y))) if family=='boost' else bootstrap(y,task,seed+i+1)
            p=params if family=='boost' else replace(params,n_estimators=1,learning_rate=1.0,row_fraction=1.0)
            m=NATIVE[task](genome=genome,config=MorphTreeConfig(seed=seed+i,enable_tuning=False,enable_morphing=False,refit_full=False),params=p,backend='pure')
            # Actual existing native .fit, never a new tree implementation.
            m=self.execute_native(m,[X[j] for j in ids],[y[j] for j in ids],identity={**identity,'member':i,'ordered_sample':ids,'params':asdict(p)},cancel=check)
            check();members.append(m)
        return FamilyModel(task,members,[1/len(members)]*len(members))


def native_fit_call(estimator, X, y, cancel):
    """Delegate for a future admitted worker, not an authorization API."""
    def cancelled():
        cancel();return False
    return estimator.fit(X,y,cancel=cancelled)
