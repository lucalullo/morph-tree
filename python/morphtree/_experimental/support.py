import copy,math,json,tempfile,os
from pathlib import Path
from dataclasses import asdict,replace
from morphtree.config import AlgorithmGenome,HyperParams
from morphtree.persistence import read_json,write_json,LoadLimits
from morphtree.model import MorphTreeClassifier,MorphTreeRegressor,MorphTreeMulticlassClassifier
from morphtree.metrics import point_losses
from .admission import digest,sha,AdmissionError
from .runtime_io import artifact_path,configure_temp,reference,publish
NATIVE={'binary':MorphTreeClassifier,'multiclass':MorphTreeMulticlassClassifier,'regression':MorphTreeRegressor}
TASKS=('binary','multiclass','regression')
def point_risks(task, y, predictions):
    if len(y) != len(predictions) or not y:
        raise ValueError('Aligned nonempty risks required')
    if task == 'binary':
        return point_losses(y, predictions)
    if task == 'regression':
        values = [(a - b) ** 2 for a, b in zip(y, predictions)]
    elif task == 'multiclass':
        values = [-math.log(max(1e-15, row[label])) for row, label in zip(predictions, y)]
    else:
        raise ValueError('Known task required')
    if any((not math.isfinite(v) for v in values)):
        raise ValueError('Finite predictions required')
    return values

def average(task, columns, weights):
    if not columns or len(columns) != len(weights) or abs(math.fsum(weights) - 1) > 1e-12:
        raise ValueError('Normalized ensemble required')
    if task == 'multiclass':
        return [[math.fsum((w * p[i][j] for w, p in zip(weights, columns))) for j in range(len(columns[0][i]))] for i in range(len(columns[0]))]
    return [math.fsum((w * p[i] for w, p in zip(weights, columns))) for i in range(len(columns[0]))]

class FamilyModel:

    def __init__(self, task, components, weights):
        self.task, self.components, self.weights = (task, components, weights)

    def probabilities_or_values(self, X):
        columns = []
        for m in self.components:
            p = m.predict(X) if self.task == 'regression' else m.predict_proba(X)
            if self.task == 'binary':
                p = [row[1] for row in p]
            columns.append(p)
        return average(self.task, columns, self.weights)

    def predict(self, X):
        p = self.probabilities_or_values(X)
        if self.task == 'regression':
            return p
        if self.task == 'binary':
            return [int(v >= 0.5) for v in p]
        return [max(range(len(v)), key=v.__getitem__) for v in p]
def prediction_contract(task, y, values):
    if len(y) != len(values) or not y:
        raise ValueError('Aligned predictions required')
    if task == 'multiclass':
        classes = sorted(set(y))
        if classes != list(range(len(classes))):
            raise ValueError('Explicit contiguous class vocabulary required')
        if any((len(p) != len(classes) or any((not math.isfinite(q) or q < 0 for q in p)) or abs(math.fsum(p) - 1) > 1e-12 for p in values)):
            raise ValueError('Aligned finite probability vectors required')
    elif any((type(p) not in (int, float) or not math.isfinite(p) or (task == 'binary' and (not 0 <= p <= 1)) for p in values)):
        raise ValueError('Finite task-appropriate predictions required')
    return point_risks(task, y, values)
def primary(task, y, prediction):
    losses = prediction_contract(task, y, prediction)
    return math.fsum(losses) / len(losses)

class FrozenFinalizer:
    """One fixed form, explicit numeric opportunity, retain on ties/failure.

    verify(packet, SEARCH) must authenticate completed model requests and return
    predictions from those exact artifacts. It may load, never silently refit.
    No GUARD or DEV argument exists. Recipe contains weights/member layout/map/
    operator; numeric fields cannot alter it. Callback exceptions propagate and
    leave the previously completed incumbent intact, never a partial result.
    """

    def __init__(self, anchor, search, *, numeric_fields, max_trials, verify):
        if type(max_trials) is not int or not 0 <= max_trials <= 8:
            raise ValueError('Finite trial ceiling 0..8')
        if search.get('role') != 'SEARCH' or set(search) != {'role', 'ids', 'X', 'y'}:
            raise ValueError('Explicit SEARCH only')
        if not search['ids'] or len(set(search['ids'])) != len(search['ids']) or len(search['ids']) != len(search['y']) or (len(search['X']) != len(search['y'])):
            raise ValueError('Aligned unique SEARCH rows')
        if not callable(verify) or not set(numeric_fields) <= set(anchor['params']) or 'n_estimators' in numeric_fields:
            raise ValueError('Explicit verifier and numerical fields; tree budget frozen')
        self.search = copy.deepcopy(search)
        self.numeric_fields = frozenset(numeric_fields)
        self.max_trials = max_trials
        self.verify = verify
        self.anchor = copy.deepcopy(anchor)
        self.signature = self._signature(self.anchor)
        self.trace = []
        self.attempts = 0
        self.incumbent = self._verified(self.anchor)
        self.anchor_loss = self.incumbent['loss']

    def _signature(self, packet):
        return digest({k: packet[k] for k in ('task', 'family', 'recipe', 'learning_identity')} | {'fixed_params': {k: v for k, v in packet['params'].items() if k not in self.numeric_fields}})

    def _verified(self, packet):
        packet = copy.deepcopy(packet)
        if self._signature(packet) != self.signature or packet['family'] not in ('boost', 'bag', 'blend'):
            raise ValueError('Frozen family/recipe/data/source/budget changed')
        if packet['task'] not in ('binary', 'multiclass', 'regression') or packet['learning_identity']['SEARCH'] != digest(self.search):
            raise ValueError('Task or SEARCH identity mismatch')
        predictions = self.verify(copy.deepcopy(packet), copy.deepcopy(self.search))
        loss = primary(packet['task'], self.search['y'], predictions)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Finite nonnegative primary risk')
        return {'packet': packet, 'prediction': copy.deepcopy(predictions), 'loss': loss}

    def offer(self, packet):
        if self.attempts >= self.max_trials:
            raise ValueError('Trial ceiling exhausted; no renewal')
        self.attempts += 1
        candidate = self._verified(packet)
        if candidate['loss'] < self.incumbent['loss']:
            self.incumbent = candidate
        self.trace.append({'offered': packet['id'], 'SEARCH_primary': candidate['loss'], 'retained': self.incumbent['packet']['id']})
        return copy.deepcopy(self.incumbent)

    def run(self, propose):
        """Actual prospective trial loop; the anchor never becomes a default.

        A qualified callback may perform a fit outside this module. The frozen
        anchor packet is supplied by value for exact seed/recipe propagation.
        This method grants no fit budget or admission by itself.
        """
        while self.attempts < self.max_trials:
            packet = propose(self.attempts, copy.deepcopy(self.anchor))
            self.offer(packet)
        result = self._verified(self.incumbent['packet'])
        if result['prediction'] != self.incumbent['prediction'] or result['loss'] > self.anchor_loss:
            raise ValueError('Nontransactional final handoff or lost anchor')
        return copy.deepcopy(result)
class TransactionalFinalizer(FrozenFinalizer):
    """Consume an opportunity durably BEFORE a callback can start learning.

    A callback/verifier/checkpoint error poisons this instance. A new instance
    cannot reconcile the outer worker's incomplete INTENT; the existing durable
    unit/lease boundary remains mandatory. checkpoint is a trusted source-bound
    synchronous writer, never an inference provider.
    """

    def __init__(self, *args, checkpoint, **kwargs):
        if not callable(checkpoint):
            raise ValueError('Synchronous checkpoint required')
        self.checkpoint = checkpoint
        self.failed = False
        super().__init__(*args, **kwargs)

    def _begin(self):
        if self.failed or self.attempts >= self.max_trials:
            raise RuntimeError('Terminal or exhausted finalizer; reconcile, never retry')
        number = self.attempts
        self.attempts += 1
        event = {'status': 'INTENT', 'trial': number, 'anchor': self.anchor['id'], 'signature': self.signature}
        self.trace.append(event)
        try:
            self.checkpoint(copy.deepcopy(event))
        except BaseException:
            self.failed = True
            raise
        return number

    def _complete(self, packet):
        candidate = self._verified(packet)
        if candidate['loss'] < self.incumbent['loss']:
            self.incumbent = candidate
        event = {'status': 'COMPLETED', 'trial': self.attempts - 1, 'offered': packet['id'], 'SEARCH_primary': candidate['loss'], 'retained': self.incumbent['packet']['id']}
        self.checkpoint(copy.deepcopy(event))
        self.trace.append(event)
        return copy.deepcopy(self.incumbent)

    def offer(self, packet):
        self._begin()
        try:
            return self._complete(packet)
        except BaseException:
            self.failed = True
            raise

    def run(self, propose):
        if self.failed:
            raise RuntimeError('Failed callback cannot replay')
        while self.attempts < self.max_trials:
            number = self._begin()
            try:
                self._complete(propose(number, copy.deepcopy(self.anchor)))
            except BaseException:
                self.failed = True
                raise
        try:
            result = self._verified(self.incumbent['packet'])
            if result['prediction'] != self.incumbent['prediction'] or result['loss'] > self.anchor_loss:
                raise ValueError('Changed final artifact or lost anchor')
            return copy.deepcopy(result)
        except BaseException:
            self.failed = True
            raise
def risk(task, y, prediction, classes):
    if task not in TASKS or not y or len(y) != len(prediction):
        raise ValueError('Aligned finite task risk')
    if task != 'multiclass':
        if any((type(p) not in (int, float) or not math.isfinite(p) or (task == 'binary' and (not 0 <= p <= 1)) for p in prediction)):
            raise ValueError('Finite native predictions')
        values = point_risks(task, y, prediction)
    else:
        if type(classes) is not list or not classes or len(set(classes)) != len(classes):
            raise ValueError('Explicit unique TRAIN class vocabulary')
        mapping = {v: i for i, v in enumerate(classes)}
        if any((v not in mapping for v in y)):
            raise ValueError('Unknown SEARCH label')
        if any((type(p) is not list or len(p) != len(classes) or any((type(q) not in (int, float) or not math.isfinite(q) or (not 0 <= q <= 1) for q in p)) or (abs(math.fsum(p) - 1) > 1e-12) for p in prediction)):
            raise ValueError('TRAIN-aligned finite probability vectors')
        values = [-math.log(max(1e-15, p[mapping[v]])) for v, p in zip(y, prediction)]
    value = math.fsum(values) / len(values)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Nonnegative finite risk')
    return value

class TaskFinalizer(TransactionalFinalizer):
    """Reuse the poisoned/durable coordinator; bind full TRAIN class columns."""

    def _verified(self, value):
        p = copy.deepcopy(value)
        if self._signature(p) != self.signature or p['family'] not in ('boost', 'bag') or p['task'] not in TASKS or (p['learning_identity']['SEARCH'] != digest(self.search)):
            raise ValueError('Frozen task/form/data changed')
        pred = self.verify(copy.deepcopy(p), copy.deepcopy(self.search))
        return {'packet': p, 'prediction': copy.deepcopy(pred), 'loss': risk(p['task'], self.search['y'], pred, p['recipe']['classes'])}

class FamilyChoice:
    """One static post-evaluation family comparison, never a predictive policy.

    All source/model acquisition costs belong to the caller. Failed or consumed
    instances cannot be replayed; the verified anchor artifact remains separate.
    """

    def __init__(self, anchor, selection, *, checkpoint, verify=None):
        if not callable(checkpoint) or not callable(verify) or selection.get('role') != 'SEARCH':
            raise ValueError('SEARCH and synchronous verifier/journal required')
        self.anchor = copy.deepcopy(anchor)
        self.selection = copy.deepcopy(selection)
        self.verify = verify
        self.checkpoint = checkpoint
        self.failed = False
        self.done = False
        self.attempts = 0
        self.best = self._checked(self.anchor)
        self.anchor_loss = self.best['loss']

    def _checked(self, p):
        if p['task'] != self.anchor['task'] or p['family'] not in ('boost', 'bag') or p['learning_identity'] != self.anchor['learning_identity']:
            raise ValueError('Common actual TRAIN/SEARCH/task required')
        pred = self.verify(copy.deepcopy(p), copy.deepcopy(self.selection))
        return {'packet': copy.deepcopy(p), 'prediction': copy.deepcopy(pred), 'loss': risk(p['task'], self.selection['y'], pred, p['recipe']['classes'])}

    def run(self, offer):
        if self.failed or self.done:
            raise RuntimeError('Terminal family decision; no replay')
        try:
            if offer['family'] == self.anchor['family']:
                raise ValueError('Other native family only; numeric search is separate')
            self.attempts += 1
            self.checkpoint({'status': 'INTENT', 'offer': offer['id'], 'actual_anchor': self.anchor['id']})
            candidate = self._checked(offer)
            if candidate['loss'] < self.best['loss']:
                self.best = candidate
            self.checkpoint({'status': 'COMPLETED', 'retained': self.best['packet']['id'], 'loss': self.best['loss']})
            final = self._checked(self.best['packet'])
            if final['prediction'] != self.best['prediction'] or final['loss'] > self.anchor_loss:
                raise ValueError('Changed final artifact or lost anchor')
            self.done = True
            return {**final, 'anchor_loss': self.anchor_loss, 'actual_anchor': self.anchor['id'], 'policy': 'POSTHOC_SEARCH_BOTH_FAMILIES_ALREADY_ACQUIRED', 'predictive_policy_utility_demonstrated': False}
        except BaseException:
            self.failed = True
            raise
