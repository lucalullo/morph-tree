"""Installed COMP002 hooks for the existing LearningControl, not a new learner.

Copied into the private wheel with relative imports. No import of this file as
workspace runtime and no implicit authority from the public fit API.
"""
import copy
from dataclasses import asdict
from morphtree.config import AlgorithmGenome
from morphtree.mutations import propose_mutations
from .admission import AdmissionError, digest
from .product import MorphTree as Consumer
from .learning import LearningControl, mutation_admission
from .support import FamilyChoice, TaskFinalizer
from .execution_v2 import ACTIVE, require_session, SearchBudgetExhausted


def learning_program(packet):
    """Bounded native procedure, separate from numbers, weights and lineage.

    The instructions are derived from the existing executable genome, never
    arbitrary Python. This is the current native grammar, not a new general
    programming language or evidence that every primitive is scientifically useful.
    """
    task, family = packet['task'], packet['family']
    if task not in ('binary', 'multiclass', 'regression') or family not in ('boost', 'bag'):
        raise ValueError('Existing native task/family required')
    genome = AlgorithmGenome(**packet['recipe']['genome'])
    if task != 'binary' and genome != AlgorithmGenome():
        # Only the separately versioned native capability may admit this.
        # Consolidated/older runtimes keep rejecting nonbinary transformations.
        from morphtree import config as native_config
        validate=getattr(native_config,'validate_task_genome',None)
        if not callable(validate):
            raise ValueError('Nonbinary structural dispatch is not qualified')
        validate(task,genome)
    body={'schema':'comp008-native-learning-program-v2', 'version':2, 'task':task,
        'family':family, 'genome':genome.to_dict(),
        'instructions':[
            {'op':'preprocess', 'implementation':'native.TRAIN_only_encoder'},
            {'op':'learning', 'implementation':'native.sequential_derivatives' if family=='boost'
                else 'native.bootstrap_independent_one_stage'},
            {'op':'tree_growth', 'implementation':genome.growth_policy},
            {'op':'split_score', 'stack':list(genome.split_program)},
            {'op':'leaf_update', 'implementation':genome.leaf_policy},
            {'op':'prediction', 'implementation':'native.stage_sum' if family=='boost' else 'native.member_mean'}],
        'execution':'EXISTING_NATIVE_CONSTRUCTORS', 'code_evaluation':'NONE',
        'grammar':'EXISTING_ALGORITHM_GENOME_NATIVE_PRIMITIVES',
        'sampling':'native.configured_row_sampling' if family=='boost' else 'native.bootstrap',
        'state_update':'native.sequential_boosting' if family=='boost' else 'native.independent_member_append',
        'termination':'FIXED_NATIVE_LEARNING_BUDGET_AND_COOPERATIVE_CANCEL'}
    return dict(body,program_id='mt-program:'+digest(body))


def program_dispatch(program):
    """Validate all instructions against their actual native dispatch."""
    packet={'task':program['task'], 'family':program['family'],
            'recipe':{'genome':program['genome']}}
    if program != learning_program(packet):
        raise ValueError('Program instructions do not describe the native implementation')
    return program['family'], AlgorithmGenome(**program['genome'])


def transformed_program(parent, proposal):
    family, genome = program_dispatch(parent)
    changed=[k for k,v in genome.to_dict().items() if proposal.genome.to_dict()[k]!=v]
    if changed != [proposal.gene]:
        raise ValueError('One actual typed native gene must change, not a numeric alias')
    return learning_program({'task':parent['task'],'family':family,
                             'recipe':{'genome':proposal.genome.to_dict()}})


def evolution_context(context, selected, program, *, authenticate):
    """Validate information timing; authentication stays with the owned producer.

    An OOF label, hash or fold declaration alone cannot authenticate a producer.
    Current installed sessions have no such producer and therefore abstain. A
    later LIVE binding needs its own actual fold-artifact qualification.
    """
    required={'role','parent_packet_sha256','program_sha256','learning_identity',
              'profile','allowed_programs','evidence'}
    if type(context) is not dict or set(context)!=required:
        raise ValueError('Exact internal diagnostic context required')
    if (context['role']!='TRAIN_OOF' or context['parent_packet_sha256']!=digest(selected)
            or context['program_sha256']!=digest(program)
            or context['learning_identity']!=selected['learning_identity']):
        raise ValueError('Only current fixed-parent TRAIN OOF may guide a transformation')
    if not callable(authenticate) or authenticate(copy.deepcopy(context),copy.deepcopy(selected)) is not True:
        raise ValueError('Actual OOF producer/artifacts have not been authenticated')
    rows=context['profile'].get('diagnoses')
    if type(rows) is not list or len(rows)>32 or len({r['name'] for r in rows})!=len(rows):
        raise ValueError('Bounded distinct diagnostic hypotheses required')
    for r in rows:
        if not isinstance(r['name'],str) or any(type(r.get(k)) not in (int,float) or not 0<=r[k]<=1 for k in ('strength','confidence')):
            raise ValueError('Finite bounded diagnostic strengths/confidences')
    allowed=context['allowed_programs']
    if type(allowed) is not list or len(allowed)>64 or len(set(allowed))!=len(allowed):
        raise ValueError('Explicit bounded admitted program identities required')
    return context


class StructuralChoice(FamilyChoice):
    """Same coordinator checks/strict loss ordering, one typed growth proposal.

    FamilyChoice's family-change precondition is deliberately replaced only
    for this versioned structural case. Verification and anchor semantics stay
    those of the original coordinator.
    """
    def offer_growth(self, candidate):
        if self.failed or self.done:raise RuntimeError('Terminal structural decision; no replay')
        before = copy.deepcopy(self.best)
        old = before['packet']; new = candidate
        if new['task'] != 'binary' or old['task'] != 'binary':
            raise ValueError('Existing binary growth operator only')
        if new['recipe']['genome'] != AlgorithmGenome(growth_policy='depthwise').to_dict():
            raise ValueError('Exact existing depthwise rule required')
        try:
            checked = self._checked(candidate)
            if checked['loss'] < before['loss']:
                self.best = checked
            # Recheck retained predictions, including a candidate that lost.
            again = self._checked(self.best['packet'])
            if again != self.best: raise ValueError('Retained anchor changed')
            self.checkpoint({'phase': 'STRUCTURAL', 'status': 'COMPLETED',
                'retained': self.best['packet']['id'], 'candidate': new['id'],
                'decision': 'ACCEPT' if self.best['packet']['id']==new['id'] else 'ROLLBACK'})
            self.done=True
            return self.best
        except BaseException:
            self.best = before
            self.failed=True
            raise

    def offer_transform(self, candidate, program):
        """Reuse original verified anchor/loss semantics for a typed child.

        A new instance is used for each step, anchored to the current retained
        weights. Strict ties retain the parent. No in-place update is claimed.
        """
        if self.failed or self.done:raise RuntimeError('Terminal structural decision')
        before=copy.deepcopy(self.best)
        try:
            if (candidate['family']!=self.anchor['family']
                    or candidate['params']!=self.anchor['params']
                    or candidate['recipe']['classes']!=self.anchor['recipe']['classes']
                    or candidate['recipe']['encoder']!=self.anchor['recipe']['encoder']
                    or learning_program(candidate)!=program
                    or learning_program(self.anchor)==program):
                raise ValueError('Child must execute the proposed program with frozen numbers/data')
            checked=self._checked(candidate)
            if checked['loss']<before['loss']:self.best=checked
            if self._checked(self.best['packet'])!=self.best:
                raise ValueError('Retained parent/child predictions changed')
            self.checkpoint({'phase':'EVOLUTION','status':'COMPLETED',
                'parent':self.anchor['id'],'candidate':candidate['id'],
                'retained':self.best['packet']['id'],
                'decision':'ACCEPT' if self.best['packet']['id']==candidate['id'] else 'ROLLBACK'})
            self.done=True
            return self.best
        except BaseException:
            self.best=before;self.failed=True
            raise


class MorphTree(Consumer):
    """Original orchestration reached through the installed canonical API.

    An exact current session is mandatory. A software session only supplies
    explicitly constructed/retained weights; PREFIT stops before native work.
    """
    _numeric_plan = LearningControl._numeric_plan
    _log = LearningControl._log

    def __init__(self, *, task='auto', families='boost', morphing='none', tuning='none', **kwargs):
        super().__init__(task=task, families=families, morphing=morphing, tuning=tuning, **kwargs)
        self._research_growth = False
        self._evolving_program = False  # New public source opts in; old installed v6 is preserved.
        self._max_mutations = 2

    @classmethod
    def baseline(cls, *, family='boost', **kwargs):
        if family not in ('boost', 'bag') or set(kwargs) & {'families', 'morphing'}:
            raise ValueError('Baseline requires one native family and no structural search')
        return cls(families=family, morphing='none', **kwargs)

    @classmethod
    def adaptive(cls, **kwargs):
        if set(kwargs) & {'families', 'morphing'}:
            raise ValueError('Adaptive uses common native family exploration and diagnostic abstention')
        return cls(families='auto', morphing='auto', **kwargs)

    @classmethod
    def research(cls, *, operators, **kwargs):
        if tuple(operators) != ('growth_policy:depthwise',):
            raise ValueError('Explicit qualified research operator required')
        obj = cls.adaptive(**kwargs)
        obj._research_growth = True
        return obj

    @property
    def mode(self):
        if self._research_growth:
            return 'research'
        return 'adaptive' if self.families == 'auto' or self.morphing == 'auto' else 'baseline'

    def _make_executor(self):
        return require_session().executor(self)

    def _execution_check(self):
        require_session()  # require_session already checks the same Session.

    def _search_budget_open(self):
        s=ACTIVE.get()
        return s is None or s.mode!='PUBLIC_USER' or not s._search_closed

    def _close_search_budget(self,reason):
        s=ACTIVE.get()
        if s is None or s.mode!='PUBLIC_USER':raise PermissionError('No ordinary search scope')
        s.finish_search(reason)

    def _finish_search_if_public(self,reason):
        s=ACTIVE.get()
        if s is not None and s.mode=='PUBLIC_USER':s.finish_search(reason)

    def _begin_numeric_window(self):
        s=ACTIVE.get()
        if s is not None and s.mode=='PUBLIC_USER':
            if s._search_deadline is None:s.begin_search()
            else:s.check()  # No renewal after family/structural work.

    def _admit_learning_batch(self,task,family,params,kind):
        s=require_session()
        if s.mode=='PUBLIC_USER':s.admit_batch(task,family,asdict(params),kind=kind)

    def _budget_stopped(self,stage,anchor,attempt,log,error):
        self._close_search_budget('SEARCH_BUDGET_'+stage)
        attempt['search_budget_stop']={'stage':stage,'retained':anchor['id'],
                                      'reason':str(error),'tuning_complete':False}
        log('search_budget_stopped',**copy.deepcopy(attempt['search_budget_stop']))

    def _record_event(self, event):
        require_session().event('ORCHESTRATION', value=event)

    def _capture_report_packet(self, packet, model, selection, train):
        """Retain actual executed objects for post-fit verification, not authority."""
        self._report_evidence[packet['id']] = (copy.deepcopy(packet), model,
                                               copy.deepcopy(selection), copy.deepcopy(train))

    def _verify_evolution_report(self, *, finalization=None):
        from .evolution_report import verified_evolution_report
        from .base import _payloads
        from .support import risk
        session=require_session()
        from . import component_cache_v2 as cache
        from .runtime_io import artifact_path
        from .execution_v2 import validate_model
        from .support import FamilyModel
        snapshots=None
        def components():
            # Projection-local snapshot only. No reusable runtime/cache trust.
            # Decode each completed component once,then recheck matching receipt,
            # artifact hash and provenance on every packet/final verification.
            nonlocal snapshots
            if snapshots is None:
                snapshots={}
                for path in session.store.keys.glob('*/COMPLETE.json'):
                    session.check();row=cache._read(path)
                    request=cache.FitRequest(cache.canonical(row['request']))
                    receipt=session.store._result(request,path.parent)
                    native=validate_model(artifact_path(receipt['artifact']['path']),request.value())
                    normalized=_payloads(FamilyModel(row['request']['task'],[native],[1.]))[0]
                    snapshots[request.key]=(path,request,receipt,normalized)
            return snapshots
        def verify(packet):
            p,model,search,train=self._report_evidence[packet['id']]
            if p!=packet or digest(_payloads(model))!=packet['artifact_sha256']:
                raise ValueError('Report refers to altered native weights or packet')
            # Request keys are bound to exactly these native payloads by the
            # committed store and executor. Reopen each receipt, do not count
            # a family packet as one physical bagging member.
            keys=[]
            payloads=_payloads(model)
            for path,request,receipt,normalized in components().values():
                v=request.value()
                if (normalized in payloads and v['task']==packet['task'] and v['rows']==len(train['y'])
                        and v['family']==packet['family']
                        and v['structure']==packet['recipe']['genome']):
                    session.check()
                    if session.store._result(request,path.parent)!=receipt:
                        raise ValueError('Report receipt/provenance changed after snapshot')
                    keys.append(request.key)
            if len(keys)!=len(payloads):raise ValueError('Missing committed report components')
            return {'packet_sha256':digest(packet),'component_keys':keys,
                'prediction':model.probabilities_or_values(search['X']),
                'TRAIN_primary':risk(packet['task'],train['y'],model.probabilities_or_values(train['X']),packet['recipe']['classes']),
                'artifacts':{'native_payload_sha256':packet['artifact_sha256']}}
        def oof(row):
            p=session._diagnostic_proofs.get(row['context'])
            if p is None or not session.verify_evolution_context(p['context'],p['parent']):
                raise ValueError('Missing authenticated OOF proof for report')
            from . import component_cache_v2 as cache
            from .runtime_io import artifact_path
            proof=cache._read(artifact_path(p['artifact']))
            receipts=[x for f in proof['folds'] for x in f['receipts']]
            keys={cache.FitRequest(cache.canonical(x['request'])).key for x in receipts}
            cached=sum(r['event']=='CACHE_HIT' and r.get('key') in keys for r in session.store.events())
            return {'context':row['context'],'parent':digest(p['parent']),
                'program_sha256':p['context']['program_sha256'],'components':len(receipts),
                'cache_components':cached,'mode':proof['mode'],'evidence':p['context']['evidence'],
                'seconds':proof['acquisition_seconds_operational'],
                'component_keys':sorted(keys),
                'artifacts':{'path':p['artifact'],'sha256':p['sha256']}}
        packets={n:p[0] for n,p in self._report_evidence.items()}
        def metric(packet,pred):
            search=self._report_evidence[packet['id']][2]
            return risk(packet['task'],search['y'],pred,packet['recipe']['classes'])
        final=self.last_attempt_['best_verified_anchor']
        def unadopted(keys):
            if session.mode!='PUBLIC_USER' or not self.last_attempt_.get('search_budget_stop'):
                raise ValueError('Only explicit public typed-cutoff recovery may account unadopted work')
            rows={}
            for key in keys:
                session.check();path=artifact_path(session.store.keys/key/'INTENT.json')
                intent=cache._read(path);request=cache.FitRequest(cache.canonical(intent['request']))
                if request.key!=key or intent['key']!=key:raise ValueError('Invalid incomplete component intent')
                committed=key in components();artifact=None
                if committed:
                    cp,r,receipt,_=components()[key]
                    if r!=request or session.store._result(r,cp.parent)!=receipt:
                        raise ValueError('Unadopted completed component changed')
                    artifact=copy.deepcopy(receipt['artifact'])
                rows[key]={'key':key,'intent_sha256':cache.sha(path),
                    'committed':committed,'artifact':artifact,'used_in_final_model':False}
            return rows
        projection=verified_evolution_report(session.store.events(),packets=packets,
            verify_packet=verify,verify_oof=oof,program=learning_program,metric=metric,
            final_packet=final,mode=session.mode,finalization=finalization,
            budget_recovery=unadopted if session.mode=='PUBLIC_USER' else None)
        projection['component_audit']={'native_decodes':len(snapshots or {}),
            'scope':'ONE_DECODE_PER_COMMITTED_COMPONENT_PER_PROJECTION',
            'integrity':'Matching receipt/artifact/provenance rechecked on every packet and final verification;no cross-projection cache'}
        m=self.report_['morphing'];m['verified_report']=projection
        m.update(diagnosis_executed=projection['diagnosis_executed'],
            decision='EVOLUTION_EVALUATED' if projection['steps'] else 'ABSTAIN_NO_ADMITTED_TRANSFORMATION',
            information_acquisition={'new_OOF':projection['counts']['OOF_physical_fits_completed'],
                'new_fits':projection['counts']['OOF_physical_fits_completed'],
                'measured_seconds':projection['OOF_acquisition_method_seconds'],
                'cost_scope':projection['cost_scope']})
        if projection.get('budget_recovery'):
            m['decision']='PARTIAL_SEARCH_BUDGET_VERIFIED_ANCHOR_RETURNED'
            m['information_acquisition']['complete_contexts']=projection['counts']['OOF_contexts_acquired']
            m['information_acquisition']['OOF_information_status']='Only committed fold proofs are diagnostic contexts;partial native OOF work is unadopted'
        self.report_['limitations']=[
            'No independent adaptive utility/performance/platform qualification',
            'Child FULL_RETRAIN; no in-place warm start',
            'Training export is an executor-bound fragment, not autonomous search',
            'Ordinary user-data execution is not a scientific source/owner permit' if session.mode=='PUBLIC_USER' else 'General public training outside exact owned admission is blocked']
        self.orchestration_report_=copy.deepcopy(self.report_)
        return projection

    def _parameters(self, params):
        return require_session().parameters(params)

    def _structure_packet(self, genome, classes, members, encoder):
        session = require_session()
        lineage = session.packet_lineage(genome) if hasattr(session, 'packet_lineage') else {}
        if not lineage and genome!=AlgorithmGenome() and hasattr(self,'_active_evolution_parent'):
            parent=self._active_evolution_parent
            lineage={'gene':'growth_policy','parent_packet_sha256':digest(parent),
                'from':copy.deepcopy(parent['recipe']['genome']),'to':genome.to_dict(),
                'reason':'OOF_HEURISTIC_HYPOTHESIS_NOT_CAUSAL_UTILITY'}
        return {'genome':genome.to_dict(), 'classes':list(range(len(classes))),
            'members':members, 'encoder':encoder.to_dict(), 'lineage':lineage}

    def _extra_opportunities(self):
        if self._evolving_program and self.morphing=='auto':return self._max_mutations
        return int(require_session().structural and self.morphing=='auto' and self._research_growth)

    def _optional_optuna(self):
        return require_session().optional_optuna()

    def _numeric_admitted(self, executor):
        return require_session().numeric_admitted(executor)

    def _load_optuna(self):
        return require_session().load_optuna()

    def _diagnose(self, context):
        if context != {'OOF':None, 'admitted_operators':[]}:
            raise ValueError('No caller-supplied OOF/GUARD in this profile')
        result = mutation_admission(self.morphing)
        if self.morphing == 'auto':
            result.update(diagnosis_status='ABSTAIN_NO_AUTHENTIC_FIXED_TRAIN_OOF',
                          decision='ABSTAIN_DIAGNOSTIC_GUIDANCE', utility='INCONCLUSIVE')
        return result

    def _structural_stage(self, selected, result, acquire, verify, selection, params, log, attempt):
        if self._evolving_program:
            try:return self._evolve_program(selected,result,acquire,verify,selection,params,log,attempt)
            except SearchBudgetExhausted as exc:
                anchor=copy.deepcopy(attempt['best_verified_anchor'])
                self._budget_stopped('EVOLUTION',anchor,attempt,log,exc)
                trace=attempt['evolution']
                if trace['steps'] and trace['steps'][-1]['status']=='INTERRUPTED':
                    step=trace['steps'][-1]
                    log('evolution_interrupted',index=step['index'],parent=step['parent'],
                        retained=anchor['id'],reason='TYPED_SEARCH_BUDGET',
                        decision='RESTORE_VERIFIED_PARENT')
                trace.update(stop='SEARCH_BUDGET_DURING_EVOLUTION',retained=anchor['id'],
                             accepted_program=learning_program(anchor))
                log('evolution_complete',program=learning_program(anchor),stop=trace['stop'],
                    retained=anchor['id'],utility='NOT_DEMONSTRATED')
                from .support import risk
                prediction=verify(anchor,selection)
                return anchor,{'packet':anchor,'prediction':prediction,
                    'loss':risk(anchor['task'],selection['y'],prediction,anchor['recipe']['classes'])}
        session = require_session()
        if not self._research_growth or not session.structural or self.morphing=='none' or selected['task']!='binary':
            return selected, result
        # Explicit static technical nomination, never inferred diagnostic value.
        proposals = [p for p in propose_mutations(AlgorithmGenome(), {'diagnoses':[]})
                     if p.gene=='growth_policy' and p.genome.growth_policy=='depthwise']
        if len(proposals)!=1: raise ValueError('Existing canonical operator missing/ambiguous')
        proposal = proposals[0]
        log('structural_proposal', gene=proposal.gene, genome=proposal.genome.to_dict(), parent_packet=copy.deepcopy(selected),
            origin='EXPLICIT_STATIC_TECHNICAL_NOMINATION_NOT_SELF_DIAGNOSIS')
        coordinator = StructuralChoice(selected, selection, verify=verify,
            checkpoint=lambda e:log('structural_selection', **e))
        technical={'origin':'STATIC_TECHNICAL_NOT_DIAGNOSTIC_GUIDANCE',
            'gene':proposal.gene,'parent_id':selected['id'],
            'parent_packet_sha256':digest(selected),'candidate_id':'growth:depthwise',
            'candidate_genome':proposal.genome.to_dict(),'status':'INTENT'}
        attempt['structural']=technical
        try:
            # The selected family is preserved; bag members reset just as in Base.
            candidate = acquire('growth:depthwise', selected['family'], params, proposal.genome)
            result = coordinator.offer_growth(candidate)
            selected = result['packet']
            attempt['best_verified_anchor'] = copy.deepcopy(selected)
            technical.update(status='COMPLETED',retained=selected['id'],
                decision='ACCEPT' if selected['id']=='growth:depthwise' else 'ROLLBACK',
                SEARCH_primary=result['loss'])
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception as exc:
            session.check()  # STOP/deadline must not be swallowed as a candidate failure.
            log('structural_rollback', reason=type(exc).__name__, message=str(exc),
                retained=selected['id'])
            result = coordinator.best
            technical.update(status='FAILED_ROLLED_BACK',retained=selected['id'],
                             decision='ROLLBACK',error=type(exc).__name__,message=str(exc))
        return selected, result

    def _evolve_program(self, selected, result, acquire, verify, selection, params, log, attempt):
        """Continue from accepted native programs; no static winner preselection.

        Uses the same component acquisition/coordinator. Every current child is
        explicitly a FULL_RETRAIN, not warm start. Original weights stay intact.
        Session/worker/store own physical caps, STOP and deadlines; this finite
        opportunity limit never grants learning authority or resets their caps.
        """
        session=require_session();current=learning_program(selected)
        trace={'schema':'comp007-evolution-trace-v1','initial_program':copy.deepcopy(current),
            'initial_anchor':selected['id'],'steps':[], 'diagnostics':[], 'evidence':session.mode,
            'stop':'UNKNOWN','accepted_program':copy.deepcopy(current),
            'continuation':'ACCEPTED_PROGRAM_AND_RETAINED_PARENT_WEIGHTS',
            'warm_start_implemented':False,'global_optimum_claim':False,
            'max_mutation_opportunities':self._max_mutations}
        attempt['evolution']=trace
        seen={digest(current)};attempted=0
        def stopped(reason):
            trace.update(stop=reason,retained=selected['id'])
            log('evolution_complete',program=copy.deepcopy(current),stop=reason,
                retained=selected['id'],utility='NOT_DEMONSTRATED')
            return selected,result
        if self.morphing=='none':return stopped('EXPLICIT_BASELINE')
        if selected['task']!='binary':
            from morphtree.diagnostics import ErrorProfiler
            if not callable(getattr(ErrorProfiler,'profile_task',None)):
                return stopped('ABSTAIN_NONBINARY_STRUCTURAL_DISPATCH_UNQUALIFIED')
        if not callable(getattr(session,'evolution_context',None)):
            return stopped('ABSTAIN_NO_AUTHENTIC_FIXED_PARENT_TRAIN_OOF')
        while attempted<self._max_mutations:
            if not self._search_budget_open():return stopped('SEARCH_BUDGET_BEFORE_DIAGNOSIS')
            session.check()
            context=session.evolution_context(copy.deepcopy(selected),copy.deepcopy(current))
            if context is None:
                trace['stop']='ABSTAIN_INSUFFICIENT_DIAGNOSTIC_INFORMATION';break
            context=evolution_context(context,selected,current,
                authenticate=getattr(session,'verify_evolution_context',None))
            trace['diagnostics'].append({'parent':selected['id'],'program_sha256':digest(current),
                'context_sha256':digest(context),'profile':copy.deepcopy(context['profile']),
                'evidence':context['evidence'],'causal_status':'HYPOTHESES_NOT_IDENTIFIED_CAUSES'})
            log('diagnosis_completed',parent=selected['id'],parent_packet_sha256=digest(selected),
                context_sha256=digest(context),program_sha256=digest(current),evidence=context['evidence'])
            strengths={r['name']:r['strength']*r['confidence'] for r in context['profile']['diagnoses']}
            proposals=[]
            for p in propose_mutations(AlgorithmGenome(**current['genome']),context['profile']):
                # Existing proposals are hypotheses, never automatic admission.
                try:child=transformed_program(current,p)
                except ValueError as exc:
                    log('evolution_proposal_rejected',gene=p.gene,
                        reason='INCOMPATIBLE_TYPED_PROGRAM',detail=str(exc))
                    continue  # No training or admission follows this proposal.
                key=digest(child)
                signal=p.reason.removeprefix('Hypothesis ').split(':',1)[0]
                if key not in seen and key in context['allowed_programs'] and strengths.get(signal,0)>0:
                    proposals.append((p,child,key))
            if not proposals:
                trace['stop']='NO_ADMITTED_UNTRIED_DIAGNOSTIC_TRANSFORMATION';break
            p,child,key=proposals[0];seen.add(key);attempted+=1
            before=copy.deepcopy(selected)
            step={'index':attempted,'parent':before['id'],'parent_program_sha256':digest(current),
                'parent_packet_sha256':digest(before),
                'child_program':copy.deepcopy(child),'gene':p.gene,'hypothesis':p.reason,
                'diagnostic_context_sha256':digest(context),'diagnostic_evidence':context['evidence'],
                'execution':'FULL_RETRAIN_REQUIRED_NOT_IN_PLACE','status':'INTENT',
                'numeric_parameters_sha256':digest(before['params']),
                'cost_before':copy.deepcopy(session.counters()) if callable(getattr(session,'counters',None)) else 'UNKNOWN'}
            trace['steps'].append(step);log('evolution_intent',**copy.deepcopy(step))
            coordinator=StructuralChoice(before,selection,verify=verify,
                checkpoint=lambda e:log('evolution_selection',**e))
            try:
                family,genome=program_dispatch(child)
                self._active_evolution_parent=copy.deepcopy(before)
                candidate=acquire('evolution:'+str(attempted),family,type(params)(**before['params']),genome)
                result=coordinator.offer_transform(candidate,child);selected=result['packet']
                accepted=selected['id']==candidate['id']
                step.update(status='COMPLETED',decision='ACCEPT' if accepted else 'ROLLBACK',
                            retained=selected['id'],SEARCH_primary=result['loss'])
                if accepted:current=copy.deepcopy(child)
                # Re-diagnosis next iteration is bound to THIS accepted parent.
                attempt['best_verified_anchor']=copy.deepcopy(selected)
                trace['accepted_program']=copy.deepcopy(current)
                if not accepted and set(context['allowed_programs'])<=seen:
                    # The identical parent/context cannot nominate any unseen
                    # admitted program. Do not reacquire/reverify its OOF just
                    # to rediscover the same exhausted candidate set.
                    trace['stop']='NO_UNTRIED_ALLOWED_PROGRAM_AFTER_ROLLBACK'
                    break
            except (TimeoutError,KeyboardInterrupt):
                step.update(status='INTERRUPTED',decision='RESTORE_VERIFIED_PARENT',retained=before['id'])
                raise
            except Exception as exc:
                step.update(status='FAILED',decision='ROLLBACK',retained=before['id'],error=type(exc).__name__)
                selected=before;result=coordinator.best
                session.check()  # Never swallow STOP/deadline as a mere candidate error.
                log('evolution_failure',retained=before['id'],error=type(exc).__name__)
                trace['stop']='CANDIDATE_ERROR_ROLLBACK';break
            finally:
                self.__dict__.pop('_active_evolution_parent',None)
                step['cost_after']=copy.deepcopy(session.counters()) if callable(getattr(session,'counters',None)) else 'UNKNOWN'
        else:trace['stop']='MUTATION_OPPORTUNITY_BUDGET_EXHAUSTED'
        trace['retained']=selected['id']
        log('evolution_complete',program=copy.deepcopy(current),stop=trace['stop'],
            retained=selected['id'],utility='NOT_DEMONSTRATED')
        return selected,result

    def fit(self, X, y):
        # Post-learning report verification is part of the transaction too.
        # Existing fitted objects remain recoverable if that verification fails.
        before=dict(self.__dict__)
        try:return self._fit_with_verified_reporting(X,y)
        except BaseException:
            attempt=copy.deepcopy(getattr(self,'last_attempt_',{}))
            self.__dict__.clear();self.__dict__.update(before)
            self.last_attempt_=attempt
            if attempt.get('status')=='PASS':
                self.last_attempt_.update(status='FAIL',stop_reason='REPORT_VERIFICATION_FAILED_RESTORED_PRIOR_OBJECT')
            raise

    def _fit_with_verified_reporting(self, X, y):
        session = require_session()
        session.bind_input(self, X, y)
        if session.mode=='PUBLIC_USER':session.begin_search()
        self._report_evidence = {}
        self._workdir = str(session.root)
        # Reuse the original finite fit loop (generated hooks documented in build).
        with session.orchestration_operation(self):
            result = LearningControl.fit(self, X, y)
        self._input_sha256 = None
        if session.mode=='FIXTURE' and self.report_['trials']:
            self.report_['trial_evidence']='SIMULATED_OPTUNA_FIXTURE_NOT_REAL_TRIALS'
        elif session.mode=='PUBLIC_USER' and self.report_['trials']:
            self.report_['trial_evidence']='USER_DATA_REAL_OPTUNA_NOT_SCIENTIFIC_CONFIRMATION'
        elif session.mode=='LIVE' and self.report_['trials']:
            self.report_['trial_evidence']='EXACT_OWNER_BOUND_REAL_OPTUNA_TRIALS'
        technical=self.last_attempt_.get('structural')
        if technical is not None:
            self.report_['morphing']['technical_structural_exploration']=copy.deepcopy(technical)
            self.report_['morphing']['attempted']=[technical['candidate_id']]
            key='accepted' if technical['decision']=='ACCEPT' else 'rollback'
            self.report_['morphing'][key]=[technical['candidate_id']]
            self.report_['stop_details']['structural']='FINITE_TECHNICAL_COMPARISON_'+technical['status']
        evolution=self.last_attempt_.get('evolution')
        if evolution is not None:
            self.algorithm_program_=copy.deepcopy(evolution['accepted_program'])
            self.program_history_=copy.deepcopy(evolution)
            self.report_['morphing']['evolution']=copy.deepcopy(evolution)
            if evolution['diagnostics']:
                self.report_['morphing'].update(
                    diagnosis_status='CONSUMED_INTERNAL_FIXED_PARENT_PROFILES_'+session.mode,
                    diagnostic_profiles_consumed=len(evolution['diagnostics']),
                    rediagnosis='CURRENT_RETAINED_PARENT_BOUND_CONTEXTS',utility='INCONCLUSIVE')
            self.report_['morphing']['attempted']=[s['index'] for s in evolution['steps']]
            self.report_['morphing']['accepted']=[s['index'] for s in evolution['steps'] if s.get('decision')=='ACCEPT']
            self.report_['morphing']['rollback']=[s['index'] for s in evolution['steps'] if s.get('decision')=='ROLLBACK']
            self.report_['stop_details']['structural']=evolution['stop']
        self.report_['accounting']['physical_fits']=session.physical
        self.report_['execution_mode']=self.mode
        if self.report_['tuning'] in ('OPTUNA_STOPPED_SEARCH_BUDGET','SKIPPED_SEARCH_BUDGET') or self.last_attempt_.get('search_budget_stop'):
            self.report_['stop_details']['numeric']=self.report_['tuning']
            self.report_['stop_reason']='SEARCH_BUDGET_RESERVED_FOR_FINALIZATION'
            self.stop_reason_=self.report_['stop_reason']
            self.last_attempt_['stop_reason']=self.stop_reason_
            if not self.report_['trials'] and 'tuned' in self._variants:
                self._variants.pop('tuned');self._selected_variant='pure'
        if self.last_attempt_.get('search_budget_stop'):
            self.report_['stop_details']['family']='COMPLETE' if not self.report_['families']['not_evaluated'] else 'PARTIAL_BUDGET'
        self.report_['events'].append({'operation':'FIT_TERMINATION',
            'model':'COMPLETE_VERIFIED_ANCHOR',
            'search':'PARTIAL_BUDGET' if self.last_attempt_.get('search_budget_stop') else 'CONFIGURED_PATH_COMPLETE',
            'tuning':self.report_['tuning'],'families':copy.deepcopy(self.report_['families'])})
        self.orchestration_report_ = copy.deepcopy(self.report_)
        # Existing V3 outer report is strict. Keep full cost/decisions in events;
        # do not broaden its schema or drop native/recipe state.
        allowed = {'schema','task','metric','classes','seed','best_family','SEARCH_primary',
            'selection','tuning','trials','morphing','stop_reason','worker_admission','events',
            'refit_performed','GUARD_used','final_model_role','wall_seconds',
            'process_cpu_seconds','descendant_cpu','peak_RSS','cost_scope','limitations'}
        extras = {k:v for k,v in self.report_.items() if k not in allowed}
        self.report_ = {k:v for k,v in self.report_.items() if k in allowed}
        self.report_['events'].append({'operation':'COMP003_ACCOUNTING_AND_SCOPE',
            'details':extras, 'counts':session.counters(), 'scope':session.mode})
        self._cost = session.counters()
        self.last_attempt_['physical_fits']=session.physical
        # Fixtures may be saved only under the explicitly versioned fixture
        # envelope, and retain their software evidence label after loading.
        from .producer import evolution_profile
        form='product006-frozen-form-v1' if session.mode=='PUBLIC_USER' else 'product002-frozen-form-v1'
        for variant in self._variants.values():
            variant['recipe']['schema']=form
            variant['exportable']=True
        self.best_recipe_['schema']=form
        if evolution is not None:
            self._verify_evolution_report()
            self._cost=session.counters()
        return result

    def get_algorithm_program(self):
        self._fitted()
        if not hasattr(self,'algorithm_program_'):
            raise ValueError('This historical model has no recorded evolving training program')
        return copy.deepcopy(self.algorithm_program_)

    def describe_algorithm(self):
        p=self.get_algorithm_program();history=self.program_history_
        return {'program':p,'evolution':copy.deepcopy(history),
            'training_code_export':'EXECUTOR_BOUND_TRAIN_ENCODED_FRAGMENT_NOT_STANDALONE',
            'inference_export':'USE_EXISTING_EXPORT_PYTHON',
            'utility':'INCONCLUSIVE','continuation':'FULL_RETRAIN_CHILDREN; RETAIN_PARENT_ON_ROLLBACK'}

    def export_algorithm(self,path):
        """Export a validated native training fragment, NOT autonomous training.

        The generated code requires this installed runtime and an already
        admitted/prepared Session, TRAIN encoding, identity and inspect scope.
        It neither creates a grant nor reproduces the whole evolutionary search.
        Data and generated arbitrary source never enter executable expressions.
        """
        self._io()
        from .runtime_io import artifact_path, publish
        p=self.get_algorithm_program();program_dispatch(p)
        from morphtree.config import HyperParams
        from pathlib import Path
        from .admission import sha
        HyperParams(**self.best_recipe_['params'])
        text='''# MorphTree validated native training fragment; not standalone.
# Requires an admitted installed Session and encoded TRAIN with exact provenance.
# No search, OOF, new authority or arbitrary evaluation is generated here.
from morphtree.config import HyperParams
from morphtree._experimental.integrated import program_dispatch
from morphtree._experimental.execution_v2 import require_session
PROGRAM = %r
PARAMETERS = %r
SEED = %r
RUNTIME_MANIFEST_SHA256 = %r
EXPORT_SCOPE = 'EXECUTOR_BOUND_TRAIN_ENCODED_FRAGMENT_NOT_END_TO_END_SEARCH'
def train_encoded(X, y, *, identity):
    family, genome = program_dispatch(PROGRAM)
    session = require_session()
    if session.engine.identity['manifest_sha256'] != RUNTIME_MANIFEST_SHA256:
        raise ValueError('Training fragment requires its exact admitted runtime manifest')
    return session.executor(None).fit_family(PROGRAM['task'], family, X, y,
        seed=SEED, params=HyperParams(**PARAMETERS), genome=genome,
        check=session.check, identity=identity)
''' % (p,copy.deepcopy(self.best_recipe_['params']),self.best_recipe_['seed'],
       sha(Path(__file__).with_name('manifest.json')))
        publish(artifact_path(path),text,lambda:None)

    def get_mutation_history(self):
        self.get_algorithm_program()
        return copy.deepcopy(self.program_history_)

    def get_recipe(self):
        return self.recipe()

    def get_search_report(self):
        """Verified projection when available; never relabel legacy summaries."""
        self._fitted()
        projection=self.report_.get('morphing',{}).get('verified_report')
        if projection is None:
            return {'status':'HISTORICAL_UNRECONCILED_REPORT','verified':False,
                    'report':self.report()}
        value=copy.deepcopy(projection)
        if not hasattr(self,'_report_evidence'):
            value['recorded_status']=value['status']
            value['status']='PERSISTED_REPORT_NOT_REVERIFIED'
            value['verified']=False
            value['reverification']='PERSISTED_REPORT_JOURNAL_NOT_REOPENED'
        else:value['verified']=True
        return value

    @classmethod
    def load(cls,path,*,sha256,workdir):
        obj=super().load(path,sha256=sha256,workdir=workdir)
        trace=obj.report_.get('morphing',{}).get('evolution')
        if trace is not None:
            # The report is descriptive, never fit authority. Reconstruct its
            # final program only if it describes the safely loaded native form.
            if type(trace) is not dict or trace.get('schema')!='comp007-evolution-trace-v1':
                raise ValueError('Unknown program history')
            expected=learning_program({'task':obj.task_,'family':obj.best_family_,
                                       'recipe':obj.best_recipe_['structure']})
            if trace.get('accepted_program')!=expected:
                raise ValueError('Recorded learning program differs from loaded native model')
            cap=trace.get('max_mutation_opportunities')
            if type(cap) is not int or not 0<=cap<=8 or type(trace.get('steps')) is not list or len(trace['steps'])>cap:
                raise ValueError('Invalid bounded program history')
            obj.algorithm_program_=copy.deepcopy(expected)
            obj.program_history_=copy.deepcopy(trace);obj._max_mutations=cap
        return obj

    def cost_report(self):
        if hasattr(self, '_cost'):
            return copy.deepcopy(self._cost)|{'scope':self.report_['worker_admission'],
                'performance_qualified':False,'peak_RSS':'UNKNOWN'}
        return super().cost_report()

