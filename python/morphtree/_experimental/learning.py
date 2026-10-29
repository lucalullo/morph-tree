from .execution_v2 import SearchBudgetExhausted
import copy, importlib, importlib.util, json, math, random, tempfile, time
from ..temporary import OwnedTemporaryDirectory
from pathlib import Path
from dataclasses import asdict, replace
from morphtree.config import AlgorithmGenome, HyperParams
from morphtree.persistence import read_json, write_json, LoadLimits
from .runtime_io import artifact_path, configure_temp, reference
from .admission import digest, sha
from .support import FamilyChoice, TaskFinalizer, risk
from .parameter_contract import admit_space
from .base import Base as Previous, PARAMS, _payloads
from .preprocessing import TrainEncoder, _m26_rows
from .training import NativeExecutor, TrainingNotAdmitted
from .target import detect_target, _split, _NamedRows

class BudgetExhausted(RuntimeError):
    """Finite opportunity cap; never convergence or a fresh budget."""

class StopRequested(RuntimeError):
    """Explicit STOP; never a recoverable timeout."""

class VariantUnavailable(ValueError):
    pass

def stop_reason(*, event, completed_proposals=0, improved=False, full_neighborhood_verified=False):
    """Typed operational semantics, not a fitted stagnation detector."""
    direct = {'planned': 'PLANNED_SEARCH_COMPLETE', 'no_actions': 'NO_ADMISSIBLE_CORRECTIVE_ACTION', 'budget': 'BUDGET_EXHAUSTED', 'timeout': 'TIMEOUT', 'stop': 'STOP_REQUESTED', 'error': 'ERROR_OR_INTERRUPTION', 'not_admitted': 'EXECUTION_NOT_ADMITTED'}
    if event in direct:
        return direct[event]
    if type(completed_proposals) is not int or completed_proposals < 1 or type(improved) is not bool:
        raise ValueError('A completed comparison is required before a stall claim')
    if event == 'structural' and (not improved):
        return 'STRUCTURAL_NO_IMPROVEMENT'
    if event == 'family' and (not improved) and (full_neighborhood_verified is True):
        return 'FAMILY_SEARCH_STALLED'
    raise ValueError('No evidence of the requested stall')

def mutation_admission(mode):
    """Reuse negative knowledge: no scientifically admitted automatic operator.

    This admission decision is not a diagnostic measurement. No labels, OOF,
    SEARCH or final GUARD are accepted by this function.
    """
    if mode not in ('none', 'auto'):
        raise ValueError('Only none/auto morphing admitted')
    return {'requested': mode, 'diagnosis_executed': False, 'diagnosis_status': 'OMITTED_DISABLED' if mode == 'none' else 'OMITTED_NO_ADMITTED_CORRECTION', 'information_acquisition': {'new_OOF': 0, 'new_fits': 0, 'measured_seconds': None}, 'eligible_operators': [], 'excluded_operators': [{'id': 'MORPH-013-atomic-trigger', 'reason': 'FAILED_STRONG_CONTROLS'}, {'id': 'MORPH-014-015-simple-mean', 'reason': 'FAILED_BEST_PARENT_CONTROLS'}, {'id': 'historical-lookahead', 'reason': 'NO_INDEPENDENT_DEFAULT_UTILITY'}], 'attempted': [], 'accepted': [], 'rollback': [], 'decision': 'DISABLED' if mode == 'none' else 'RETAIN_CURRENT_NO_ADMITTED_OPERATOR', 'rediagnosis': 'NOT_EXECUTED', 'utility': 'NOT_DEMONSTRATED'}

class LearningControl(Previous):
    """Private three-task automatic facade, not a production estimator.

    balanced resolves to serial, 30s, boost8/bag8, and at most four numeric
    trials ONLY with a separately qualified executor. Without such admission,
    auto skips tuning and explicit optuna refuses before learning. No automatic
    OOF, structural mutation, composition, refit or final GUARD is enabled.
    """

    def __init__(self, *, task='auto', families='auto', morphing='auto', tuning='auto', preset='balanced', metric='auto', time_budget='auto', n_trials='auto', n_jobs='auto', random_state=None, verbose=0, capacity=None):
        self.capacity = copy.deepcopy(capacity)
        super().__init__(task=task, families=families, morphing=morphing, tuning=tuning, preset=preset, metric=metric, time_budget=time_budget, n_trials=n_trials, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    def _validate(self):
        super()._validate()
        if self.capacity is not None:
            if type(self.capacity) is not dict or not set(self.capacity) <= {'max_leaves', 'max_depth'}:
                raise ValueError('capacity supports only max_leaves/max_depth; no inactive options')
            for name, value in self.capacity.items():
                if type(value) is not int or not 1 <= value <= {'max_leaves': 16, 'max_depth': 8}[name]:
                    raise ValueError('Finite integer structural capacity required')

    def get_params(self, deep=True):
        return copy.deepcopy({k: getattr(self, k) for k in (*PARAMS, 'capacity')})

    def set_params(self, **values):
        if set(values) - set(self.get_params()):
            raise ValueError('Unknown estimator parameter')
        new = type(self)(**self.get_params() | values)
        self.__dict__ = new.__dict__
        return self

    def _make_executor(self):
        return NativeExecutor()

    def _numeric_admitted(self, executor):
        return False

    def _optional_optuna(self):
        return importlib.util.find_spec('optuna') is not None

    def _load_optuna(self):
        return importlib.import_module('optuna')

    def _numeric_plan(self, executor):
        if self.tuning == 'none':
            return (None, 'DISABLED', 0)
        if not self._optional_optuna():
            if self.tuning == 'optuna':
                raise ImportError('tuning="optuna" requires optional Optuna before any learning')
            return (None, 'AUTO_SKIPPED_DEPENDENCY', 0)
        if not self._numeric_admitted(executor):
            if self.tuning == 'optuna':
                raise TrainingNotAdmitted('Explicit numeric execution needs new source-bound fit/trial admission')
            return (None, 'AUTO_SKIPPED_NUMERIC_ADMISSION', 0)
        return (self._load_optuna(), 'ADMITTED', 2 if self.n_trials == 'auto' else self.n_trials)

    def _log(self, events, start, operation, **values):
        event = {'operation': operation, 'seconds_since_start': time.monotonic() - start, **values}
        self._record_event(copy.deepcopy(event))
        events.append(event)

    def fit(self, X, y):
        self._validate()
        start = time.monotonic()
        cpu = time.process_time()
        events = []
        attempt = {'status': 'INTENT', 'events': events, 'logical_family_requests': 0, 'numeric_callback_requests': 0, 'physical_fits': 'UNKNOWN_AT_UNADMITTED_BOUNDARY', 'best_verified_anchor': None, 'stop_reason': None}
        self.last_attempt_ = attempt

        def log(operation, **values):
            self._log(events, start, operation, **values)

        def check():
            self._execution_check()
            if time.monotonic() - start >= (150.0 if self.time_budget == 'auto' else self.time_budget):
                raise TimeoutError('Finite operational deadline, not an algorithmic stall')
        try:
            executor = self._make_executor()
            optuna, tuning_status, trial_cap = self._numeric_plan(executor)
            check()
            rows, names = _m26_rows(X)
            task, targets, classes = detect_target(y, self.task)
            if len(rows) != len(targets):
                raise ValueError('Aligned X/y required')
            metric = 'mse' if task == 'regression' else 'logloss'
            if self.metric not in ('auto', metric):
                raise ValueError('Metric is inconsistent with task')
            seed = self.random_state if self.random_state is not None else random.SystemRandom().randrange(2 ** 31)
            ti, si = _split(targets, task, seed)
            log('split', TRAIN=ti, SEARCH=si, GUARD_used=False)
            encoder = TrainEncoder().fit([rows[i] for i in ti])
            encoder.state_['names'] = names
            if encoder.width_ * len(rows) > 2000000:
                raise ValueError('Expanded input exceeds finite 2000000-cell cap')
            trainX = encoder.transform(_NamedRows([rows[i] for i in ti], names))
            searchX = encoder.transform(_NamedRows([rows[i] for i in si], names))
            ty = [targets[i] for i in ti]
            sy = [targets[i] for i in si]
            selection = {'role': 'SEARCH', 'ids': ['SEARCH:' + str(i) for i in si], 'X': searchX, 'y': sy}
            identity = {'TRAIN': digest({'X': trainX, 'y': ty, 'ids': ti}), 'SEARCH': digest(selection)}
            params = HyperParams(n_estimators=2, max_leaves=8, max_depth=4, max_bins=15, min_samples_leaf=2, row_fraction=1.0)
            if self.capacity:
                params = replace(params, **self.capacity)
            params = self._parameters(params)
            families = ['boost', 'bag'] if self.families == 'auto' else [self.families]
            models = {}
            packets = {}
            completed = []
            trials = []

            def acquire(name, family, p, genome=None):
                genome = AlgorithmGenome() if genome is None else genome
                check()
                if attempt['logical_family_requests'] >= len(families) + trial_cap + self._extra_opportunities():
                    raise BudgetExhausted('Finite family/trial opportunities exhausted')
                self._admit_learning_batch(task, family, p, 'COMPLETE_FAMILY_REQUEST')
                attempt['logical_family_requests'] += 1
                log('learning_request', id=name, family=family, params=asdict(p))
                model = executor.fit_family(task, family, trainX, ty, seed=seed, params=p, check=check, genome=genome, identity={**identity, 'preprocessing': encoder.to_dict(), 'seed': seed, 'task': task, 'family': family, 'TRAIN_ids': ['ROW:' + str(i) for i in ti], 'actual_TRAIN_X': trainX, 'actual_TRAIN_y': ty})
                check()
                payload = _payloads(model)
                packet = {'id': name, 'task': task, 'family': family, 'recipe': self._structure_packet(genome, classes, len(model.components), encoder), 'learning_identity': identity, 'params': asdict(p), 'artifact_sha256': digest(payload)}
                models[name] = model
                packets[name] = packet
                completed.append(name)
                self._capture_report_packet(packet, model, selection, {'X': trainX, 'y': ty})
                log('learning_completed', id=name, boundary=executor.evidence_level, packet=copy.deepcopy(packet))
                return packet

            def verify(packet, search):
                check()
                name = packet['id']
                if packet != packets[name] or search != selection or digest(_payloads(models[name])) != packet['artifact_sha256']:
                    raise ValueError('Changed completed native model, frozen form or SEARCH')
                return models[name].probabilities_or_values(search['X'])
            anchor = acquire(families[0], families[0], params)
            chooser = FamilyChoice(anchor, selection, checkpoint=lambda e: log('family_selection', **e), verify=verify)
            attempt['best_verified_anchor'] = copy.deepcopy(chooser.best['packet'])
            try:
                result = chooser.run(acquire(families[1], families[1], params)) if len(families) > 1 else chooser.best
            except SearchBudgetExhausted as budget_error:
                self._budget_stopped('FAMILY', attempt['best_verified_anchor'], attempt, log, budget_error)
                result = chooser.best
            selected = result['packet']
            attempt['best_verified_anchor'] = copy.deepcopy(selected)
            morph = self._diagnose({'OOF': None, 'admitted_operators': []})
            log('mutation_admission', **morph)
            selected, result = self._structural_stage(selected, result, acquire, verify, selection, params, log, attempt)
            pure = copy.deepcopy(selected)
            pure_prediction = copy.deepcopy(result['prediction'])
            frozen_result = TaskFinalizer(selected, selection, numeric_fields=[], max_trials=0, verify=verify, checkpoint=lambda e: log('finalization', **e)).run(None)
            log('structure_frozen', sha256=digest(selected['recipe']), packet=copy.deepcopy(selected), numeric_budget=trial_cap)
            self._begin_numeric_window()
            try:
                if trial_cap and self._search_budget_open():
                    space = {'l2': [0.1, 10.0], 'min_samples_leaf': [1, 8]}
                    if selected['family'] == 'boost':
                        space['learning_rate'] = [0.03, 0.3]
                    space = admit_space(space, family=selected['family'], task=task, genome=AlgorithmGenome(**selected['recipe']['genome']))['space']

                    def numeric_checkpoint(e):
                        log('numeric_selection', **e)
                        if e['status'] == 'COMPLETED':
                            attempt['best_verified_anchor'] = copy.deepcopy(packets[e['retained']])
                    finalizer = TaskFinalizer(selected, selection, numeric_fields=list(space), max_trials=trial_cap, verify=verify, checkpoint=numeric_checkpoint)
                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=2))
                    frozen = digest({'family': selected['family'], 'recipe': selected['recipe'], 'fixed_params': {k: v for k, v in selected['params'].items() if k not in space}})
                    log('structure_frozen', sha256=frozen, space=space)

                    def propose(i, frozen_anchor):
                        check()
                        attempt['numeric_callback_requests'] += 1
                        self._admit_learning_batch(task, frozen_anchor['family'], HyperParams(**frozen_anchor['params']), 'NUMERIC_BEFORE_ASK')
                        trial = study.ask()
                        try:
                            values = {n: trial.suggest_int(n, *b) if n == 'min_samples_leaf' else trial.suggest_float(n, *b, log=True) for n, b in space.items()}
                            p = replace(HyperParams(**frozen_anchor['params']), **values)
                            candidate = acquire('trial:' + str(i), frozen_anchor['family'], p, AlgorithmGenome(**frozen_anchor['recipe']['genome']))
                            loss = finalizer._verified(candidate)['loss']
                            study.tell(trial, loss)
                            trials.append({'number': trial.number, 'state': 'COMPLETE', 'params': values, 'SEARCH_primary': loss})
                            return candidate
                        except BaseException as primary_error:
                            if isinstance(primary_error, SearchBudgetExhausted):
                                self._close_search_budget('NUMERIC_SEARCH_WINDOW_EXHAUSTED')
                            try:
                                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                            except BaseException as cleanup_error:
                                log('numeric_trial_cleanup_failed', number=trial.number, error=type(cleanup_error).__name__, original_error=type(primary_error).__name__, state='UNRESOLVED')
                            raise
                    selected = finalizer.run(propose)['packet']
                    attempt['best_verified_anchor'] = copy.deepcopy(selected)
                    tuning_status = 'OPTUNA_CALLBACKS_COMPLETE'
                elif trial_cap:
                    tuning_status = 'SKIPPED_SEARCH_BUDGET'
                    log('numeric_budget_skipped', retained=selected['id'])
            except SearchBudgetExhausted as budget_error:
                self._budget_stopped('NUMERIC', attempt['best_verified_anchor'], attempt, log, budget_error)
                selected = copy.deepcopy(attempt['best_verified_anchor'])
                tuning_status = 'OPTUNA_STOPPED_SEARCH_BUDGET'
                log('numeric_search_stopped', retained=selected['id'], reason=tuning_status)
            self._finish_search_if_public('SEARCH_COMPLETE')
            check()
            prediction = verify(selected, selection)
            primary = risk(task, sy, prediction, list(range(len(classes))))
            if risk(task, sy, pure_prediction, list(range(len(classes)))) < primary:
                raise ValueError('Lost verified untuned anchor')

            def recipe(p):
                return {'schema': 'morph026-frozen-form-v1', 'task': task, 'family': p['family'], 'structure': copy.deepcopy(p['recipe']), 'params': copy.deepcopy(p['params']), 'seed': seed, 'training_identity': copy.deepcopy(identity), 'preprocessing': encoder.to_dict(), 'mutations': []}
            mock = 'MOCK' in executor.evidence_level
            report = {'schema': 'morph027-research-report-v1', 'task': task, 'metric': metric, 'classes': classes, 'seed': seed, 'best_family': selected['family'], 'SEARCH_primary': primary, 'selection': 'BEST_VERIFIED_SEARCH_NOT_GENERALIZATION_GUARANTEE', 'families': {'evaluated': [packets[n]['family'] for n in completed if n in families], 'not_evaluated': [f for f in ['boost', 'bag'] if f not in [packets[n]['family'] for n in completed if n in families]], 'exclusion_reason': 'USER_FAMILY_LIMIT_OR_SEARCH_BUDGET'}, 'tuning': tuning_status, 'trials': trials, 'trial_evidence': 'MOCK_DRIVER_NOT_REAL_TRIALS' if mock else 'EXECUTOR_BOUND_NUMERIC_CALLBACKS', 'morphing': morph, 'stop_reason': stop_reason(event='planned'), 'stop_details': {'family': 'FINITE_PREDECLARED_CANDIDATES_COMPLETE_NOT_STALL', 'structural': stop_reason(event='no_actions') if self.morphing == 'auto' else 'DISABLED', 'numeric': 'PLANNED_TRIALS_COMPLETE' if trial_cap else tuning_status, 'global_optimum_claim': False}, 'budget': {'seconds': 150.0 if self.time_budget == 'auto' else self.time_budget, 'processes': 1, 'n_jobs': 1, 'max_family_requests': len(families) + trial_cap + self._extra_opportunities(), 'max_numeric_callbacks': trial_cap, 'model_bytes': 32 * 1024 ** 2, 'expanded_cells': 2000000, 'peak_RAM_limit_qualified': False, 'external_v2_learning_admission_required': True}, 'accounting': {k: attempt[k] for k in ('logical_family_requests', 'numeric_callback_requests', 'physical_fits')}, 'worker_admission': executor.evidence_level, 'events': events, 'refit_performed': False, 'GUARD_used': False, 'final_model_role': 'TRAIN_FITTED_SEARCH_SELECTED', 'wall_seconds': time.monotonic() - start, 'process_cpu_seconds': time.process_time() - cpu, 'descendant_cpu': 'UNKNOWN', 'peak_RSS': 'UNKNOWN', 'cost_scope': 'OPERATIONAL_ONLY_NOT_PERFORMANCE', 'stop_recovery': 'ERROR_OR_STOP_LEAVES_PRIOR_FITTED_OBJECT_UNCHANGED', 'limitations': ['Research import only', 'No new native E2E evidence from software replay', 'No admitted automatic mutation or OOF', 'No independent utility/performance/platform/sklearn qualification', 'No retrainable exported procedure']}
            variants = {'pure': {'model': models[pure['id']], 'recipe': recipe(pure), 'origin': 'PRE_FINAL_NUMERIC_SEARCH', 'exportable': not mock, 'evidence': executor.evidence_level, 'SEARCH_primary': risk(task, sy, pure_prediction, list(range(len(classes))))}}
            chosen = 'pure'
            if trial_cap:
                variants['tuned'] = {'model': models[selected['id']], 'recipe': recipe(selected), 'origin': 'POST_FINAL_NUMERIC_SEARCH', 'exportable': not mock, 'evidence': executor.evidence_level, 'SEARCH_primary': primary}
                chosen = 'tuned'
            learned = {'model_': models[selected['id']], 'encoder_': encoder, 'task_': task, 'n_features_in_': len(rows[0]), 'best_family_': selected['family'], 'best_recipe_': recipe(selected), 'best_params_': copy.deepcopy(selected['params']), 'structure_': copy.deepcopy(selected['recipe']), 'stop_reason_': report['stop_reason'], 'report_': report, '_variants': variants, '_selected_variant': chosen}
            if task != 'regression':
                learned['classes_'] = tuple(classes)
            if names is not None:
                learned['feature_names_in_'] = tuple(names)
            for k in list(self.__dict__):
                if k.endswith('_') and k != 'last_attempt_' or k in ('_variants', '_selected_variant', '_untuned_model', '_untuned_recipe'):
                    del self.__dict__[k]
            self.__dict__.update(learned)
            attempt.update(status='PASS', stop_reason=report['stop_reason'])
            return self
        except BaseException as exc:
            event = 'timeout' if isinstance(exc, TimeoutError) else 'stop' if isinstance(exc, StopRequested) else 'budget' if isinstance(exc, BudgetExhausted) else 'not_admitted' if isinstance(exc, TrainingNotAdmitted) else 'error'
            attempt.update(status='BLOCKED' if event == 'not_admitted' else 'STOP' if event in ('timeout', 'stop', 'budget') else 'FAIL', stop_reason=stop_reason(event=event), error=type(exc).__name__, message=str(exc))
            raise

    def _diagnose(self, context):
        if context != {'OOF': None, 'admitted_operators': []}:
            raise ValueError('No trained diagnostic producer admitted; GUARD and caller-provided evidence forbidden')
        return mutation_admission(self.morphing)

    def report(self):
        value = super().report()
        value['variants'] = self.available_variants()
        return value

    def _evaluate_mutation(self, proposal):
        raise TrainingNotAdmitted('No default structural operator or mutation budget admitted')

    def available_variants(self):
        self._fitted()
        return {k: {'origin': v['origin'], 'params': copy.deepcopy(v['recipe']['params']), 'exportable': v['exportable'], 'evidence': v['evidence'], 'SEARCH_primary': v.get('SEARCH_primary')} for k, v in self._variants.items()}

    def _variant(self, name):
        self._fitted()
        name = self._selected_variant if name == 'selected' else name
        if name not in self._variants:
            raise VariantUnavailable('Requested variant has no retained, verified weights; a separately budgeted fit may be needed')
        v = self._variants[name]
        if not v['exportable']:
            raise VariantUnavailable('Software replay cannot be exported as a newly trained pure/tuned variant')
        obj = Previous(**{k: v for k, v in self.get_params().items() if k != 'capacity'})
        for k in ('encoder_', 'task_', 'n_features_in_', 'classes_', 'feature_names_in_'):
            if hasattr(self, k):
                setattr(obj, k, copy.deepcopy(getattr(self, k)))
        obj.model_ = v['model']
        obj.best_recipe_ = copy.deepcopy(v['recipe'])
        obj.best_family_ = v['recipe']['family']
        obj.best_params_ = copy.deepcopy(v['recipe']['params'])
        obj.report_ = self.report()
        obj.report_.update(exported_variant=name, SEARCH_primary=v.get('SEARCH_primary'), best_family=obj.best_family_)
        obj.stop_reason_ = self.stop_reason_
        return obj

    def export_python(self, path, *, variant='selected'):
        self._variant(variant).export_python(path)

    def save_recipe(self, path, *, variant='selected'):
        self._variant(variant).save_recipe(path)

    def save(self, path, *, variant='all'):
        self._fitted()
        configure_temp()
        names = list(self._variants) if variant == 'all' else [self._selected_variant if variant == 'selected' else variant]
        packed = {}
        origins = {}
        with OwnedTemporaryDirectory() as tmp:
            for name in names:
                obj = self._variant(name)
                p = artifact_path(Path(tmp) / (name + '.json'))
                obj.save(p)
                packed[name] = read_json(p, LoadLimits(max_bytes=32 * 1024 ** 2))
                origins[name] = {'origin': self._variants[name]['origin'], 'evidence': self._variants[name]['evidence']}
        body = {'schema': 'morph027-native-control-v1', 'constructor': self.get_params(), 'selected_variant': self._selected_variant if self._selected_variant in names else names[0], 'variants': packed, 'origins': origins}
        if len(json.dumps(body, allow_nan=False).encode('utf8')) > 32 * 1024 ** 2:
            raise ValueError('Bounded total variant payload required')
        path = artifact_path(path)
        if path.exists():
            raise FileExistsError('Never overwrite research model evidence')
        write_json(path, body)

    @classmethod
    def load(cls, path):
        data = read_json(artifact_path(path), LoadLimits(max_bytes=32 * 1024 ** 2))
        if set(data) != {'schema', 'constructor', 'selected_variant', 'variants', 'origins'} or data['schema'] != 'morph027-native-control-v1':
            raise ValueError('Exact bounded private control schema required')
        vs = data['variants']
        origins = data['origins']
        if type(vs) is not dict or not vs or (not set(vs) <= {'pure', 'tuned'}) or (set(origins) != set(vs)) or (data['selected_variant'] not in vs):
            raise ValueError('Bounded named variants required')
        obj = cls(**data['constructor'])
        configure_temp()
        loaded = {}
        with OwnedTemporaryDirectory() as tmp:
            for name, body in vs.items():
                o = origins[name]
                if set(o) != {'origin', 'evidence'} or o['origin'] not in ('PRE_FINAL_NUMERIC_SEARCH', 'VERIFIED_B0_UNTUNED', 'POST_FINAL_NUMERIC_SEARCH') or (not isinstance(o['evidence'], str)) or ('MOCK' in o['evidence']):
                    raise ValueError('Explicit genuine variant provenance required')
                if name == 'pure' and o['origin'] == 'POST_FINAL_NUMERIC_SEARCH' or (name == 'tuned' and o['origin'] != 'POST_FINAL_NUMERIC_SEARCH'):
                    raise ValueError('Never relabel tuned weights as pure')
                p = artifact_path(Path(tmp) / (name + '.json'))
                write_json(p, body)
                v = Previous.load(p)
                if v.task_ != body['recipe']['task']:
                    raise ValueError('Changed task')
                expected = HyperParams(**v.best_params_)
                for m in v.model_.components:
                    actual = asdict(m.params_)
                    want = asdict(expected if v.best_family_ == 'boost' else replace(expected, n_estimators=1, learning_rate=1.0, row_fraction=1.0))
                    if actual != want:
                        raise ValueError('Recipe parameters do not describe the actual weights')
                if name == 'tuned' and (not v.report_.get('trials') or v.report_.get('trial_evidence') not in ('EXECUTOR_BOUND_NUMERIC_CALLBACKS', 'AUTHENTICATED_HISTORICAL_REAL_TPE4')):
                    raise ValueError('No executed numeric trajectory for tuned variant')
                loaded[name] = v
        v = loaded[data['selected_variant']]
        for other in loaded.values():
            if (other.task_, other.classes_ if hasattr(other, 'classes_') else (), other.best_recipe_['structure'], other.best_recipe_['training_identity']) != (v.task_, v.classes_ if hasattr(v, 'classes_') else (), v.best_recipe_['structure'], v.best_recipe_['training_identity']):
                raise ValueError('Variants must share the same frozen structure/data/map')
        for k in ('model_', 'encoder_', 'task_', 'n_features_in_', 'classes_', 'feature_names_in_', 'best_family_', 'best_recipe_', 'best_params_', 'stop_reason_', 'report_'):
            if hasattr(v, k):
                setattr(obj, k, getattr(v, k))
        obj.structure_ = copy.deepcopy(v.best_recipe_['structure'])
        obj._selected_variant = data['selected_variant']
        obj._variants = {k: {'model': v.model_, 'recipe': copy.deepcopy(v.best_recipe_), **origins[k], 'exportable': True, 'SEARCH_primary': v.report_.get('SEARCH_primary')} for k, v in loaded.items()}
        return obj

    def export_bundle(self, directory, *, variants=('pure',)):
        if type(variants) is not tuple or not variants or len(set(variants)) != len(variants):
            raise ValueError('Distinct variant tuple required')
        for name in variants:
            self._variant(name)
        directory = artifact_path(directory)
        if directory.exists():
            raise FileExistsError('New export directory required')
        directory.mkdir(parents=True)
        out = []
        for name in variants:
            p = directory / (name + '.json')
            self.save(p, variant=name)
            out.append(reference(p))
            p = directory / (name + '-recipe.json')
            self.save_recipe(p, variant=name)
            out.append(reference(p))
            p = directory / (name + '-predictor.py')
            self.export_python(p, variant=name)
            out.append(reference(p))
        p = directory / 'EXPORT.json'
        write_json(p, {'schema': 'morph027-inference-export-v1', 'status': 'COMPLETE', 'artifacts': out, 'retraining_procedure': 'NOT_SUPPORTED', 'new_fits': 0, 'report': self.report()})
        return reference(p)

    def export_retrainable(self, path):
        raise NotImplementedError('Only inference standalone and declarative recipe are qualified; generic retraining admission is missing')
