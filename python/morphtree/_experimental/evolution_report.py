"""One read-only projection of the existing component journal, not a journal.

Packaged with the experimental orchestrator only. Callbacks must reopen native
artifacts/proofs and reproduce predictions; a summary flag is never evidence.
Historical reports are inputs to an optional consistency check, never authority.
"""
import copy
import hashlib
import json
import math


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def verified_evolution_report(rows, *, packets, verify_packet, verify_oof,
                              program, metric, final_packet, mode,
                              claimed=None, finalization=None,budget_recovery=None):
    """Fail closed on contradictions; UNKNOWN remains explicit in partial traces.

    No authority, fitting, file mutation or second ledger is created. Canonical
    events below are views referencing exact source sequence/hash/timestamp.
    A missing historical timestamp is UNKNOWN, never the reconciliation time.
    """
    if mode not in ('LIVE', 'FIXTURE', 'PREFIT', 'PUBLIC_USER') or len(rows)>10000:
        raise ValueError('Bounded explicit evidence mode required')
    previous='0'*64
    for i,r in enumerate(rows,1):
        body={k:v for k,v in r.items() if k!='sha256'}
        if r.get('sequence')!=i or r.get('previous')!=previous or r.get('sha256')!=_digest(body):
            raise ValueError('Broken canonical journal chain')
        if type(r.get('UTC_unix')) not in (int,float) or not math.isfinite(r['UTC_unix']):
            raise ValueError('Invalid event timestamp')
        previous=r['sha256']
    events=[];forms={};contexts={};current=None;frozen=None;steps=[]
    starts={};ends={};trials={};trial_ends={};pending=None;artifact_keys=set();diagnoses=[]
    trial_numbers=set();created_trials={};budget_stop=None;interrupted_steps=[]
    def emit(kind,r,*,parent=None,status='COMPLETED',**values):
        events.append(dict(id='view:'+str(r['sequence'])+':'+kind,kind=kind,
            parent=parent,timestamp=r['UTC_unix'],status=status,
            provenance={'sequence':r['sequence'],'sha256':r['sha256']},**values))
    def checked(name):
        if name not in packets:raise ValueError('Unverifiable program packet')
        p=packets[name]
        if p['id']!=name:raise ValueError('Changed packet identity')
        proof=verify_packet(copy.deepcopy(p))
        if proof['packet_sha256']!=_digest(p):raise ValueError('Artifact differs from program packet')
        pred=proof['prediction'];loss=metric(p,pred)
        artifact_keys.update(proof['component_keys'])
        if not math.isfinite(loss):raise ValueError('Nonfinite SEARCH loss')
        return {'packet':copy.deepcopy(p),'program':program(p),'SEARCH_primary':loss,
                'TRAIN_primary':proof.get('TRAIN_primary','UNKNOWN'),
                'artifacts':proof['artifacts'],'component_keys':list(proof['component_keys']),
                'prediction_sha256':_digest(pred)}
    for r in rows:
        k=r['event'];v=r.get('value',{});op=v.get('operation')
        if k in ('NATIVE_FIT_STARTED','NATIVE_FIT_COMPLETED'):
            target=starts if k.endswith('STARTED') else ends;key=r['key']
            if key in target:raise ValueError('Duplicate physical fit event')
            if target is ends and (key not in starts or starts[key]['phase']!=r['phase']):
                raise ValueError('Completion without corresponding physical start')
            target[key]=r;emit('FIT_STARTED' if target is starts else 'FIT_COMPLETED',r,
                status='STARTED' if target is starts else 'COMPLETED',component=key,phase=r['phase'])
        elif k=='NUMERIC_TRIAL_INTENT':
            # Existing producer reserves before Optuna.ask(), so its intent has
            # no Optuna number yet. Keep the reservation distinct from the
            # number later returned by tell; never invent an executed trial.
            n=r['sequence']
            if n in trials or frozen is None:raise ValueError('Trial duplicated or before freeze')
            trials[n]=r;emit('NUMERIC_TUNING_STARTED',r,status='STARTED',real=r['real'],
                            reservation_sequence=n,number=r.get('number','UNKNOWN_UNTIL_ASK'))
        elif k=='NUMERIC_TRIAL_ASKED' and mode=='PUBLIC_USER':
            eligible=[n for n in trials if n not in created_trials and n not in trial_ends]
            if len(eligible)!=1 or type(r.get('number')) is not int or type(r.get('real')) is not bool:
                raise ValueError('Created user trial without a unique reservation')
            n=eligible[0]
            if r['real']!=trials[n]['real'] or r['number'] in {x['number'] for x in created_trials.values()}:
                raise ValueError('Changed driver evidence or repeated user trial number')
            created_trials[n]=r
            emit('NUMERIC_TRIAL_CREATED',r,number=r['number'],real=r['real'],reservation_sequence=n)
        elif k=='NUMERIC_TRIAL_COMPLETED':
            eligible=[n for n in trials if n not in trial_ends]
            if len(eligible)!=1 or r['number'] in trial_numbers:
                raise ValueError('Trial completion without matching intent')
            n=eligible[0]
            if mode=='PUBLIC_USER' and (n not in created_trials or created_trials[n]['number']!=r['number']):
                raise ValueError('Completion before an actual returned user trial')
            if r['real']!=trials[n]['real'] or trials[n].get('number',r['number'])!=r['number']:
                raise ValueError('Trial number or evidence changed')
            trial_numbers.add(r['number'])
            trial_ends[n]=r
        elif k=='OOF_CONTEXT_PRODUCED':
            proof=verify_oof(copy.deepcopy(r))
            if (proof['context']!=r['context'] or proof['parent']!=r['parent']
                    or proof['components']!=r['components'] or proof['mode']!=mode):
                raise ValueError('OOF artifacts/provenance differ')
            if r['context'] in contexts:raise ValueError('Duplicate OOF acquisition')
            contexts[r['context']]=proof
            artifact_keys.update(proof['component_keys'])
            emit('OOF_COMPLETED',r,parent=r['parent'],components=proof['components'],
                 evidence=proof['evidence'],artifacts=proof['artifacts'])
        elif k=='OOF_CONTEXT_REUSED':
            if r['context'] not in contexts or r['physical_fits']!=0:
                raise ValueError('Unverifiable reused OOF context')
        elif op=='learning_completed':
            name=v['id']
            if name in forms:raise ValueError('Duplicate completed program')
            forms[name]=checked(name);emit('PROGRAM_CREATED',r,program=forms[name]['program'],packet=name)
            if 'packet' in v and v['packet']!=packets[name]:raise ValueError('Executed packet differs')
        elif op=='family_selection' and v['status']=='COMPLETED':
            name=v['retained']
            if name not in forms:raise ValueError('Parent not executed')
            # Verify the actual metric, including initial family comparison.
            if v['loss']!=forms[name]['SEARCH_primary']:raise ValueError('Stale family metric')
            if any(f['SEARCH_primary']<forms[name]['SEARCH_primary'] for f in forms.values()):
                raise ValueError('Lost initial verified SEARCH anchor')
            current=name;emit('PROGRAM_SELECTED_AS_PARENT',r,parent=name,program=forms[name]['program'])
        elif op=='diagnosis_completed':
            if current is None:current=v['parent']
            p=packets.get(current);ctx=contexts.get(v['context_sha256'])
            if (p is None or v['parent']!=current or ctx is None
                    or v['parent_packet_sha256']!=_digest(p)
                    or ctx['parent']!=_digest(p) or ctx['program_sha256']!=_digest(program(p))):
                raise ValueError('Unverified completed diagnosis')
            diagnoses.append(v['context_sha256'])
            emit('DIAGNOSIS_COMPLETED',r,parent=current,evidence=ctx['evidence'],context=v['context_sha256'])
        elif op=='evolution_intent':
            if current is None:current=v['parent']  # single-family path; checked below
            if current not in forms or pending is not None or frozen is not None:
                raise ValueError('Invalid mutation ordering')
            p=packets[current];ctx=contexts.get(v['diagnostic_context_sha256'])
            if (v['parent']!=current or v['parent_packet_sha256']!=_digest(p)
                    or v['parent_program_sha256']!=_digest(program(p)) or ctx is None
                    or ctx['parent']!=_digest(p) or ctx['program_sha256']!=_digest(program(p))):
                raise ValueError('Wrong parent or diagnostic provenance')
            child=v['child_program'];program_packet={'task':child['task'],'family':child['family'],
                                                     'recipe':{'genome':child['genome']}}
            if program(program_packet)!=child:raise ValueError('Declared instructions differ from dispatch')
            changed=[g for g,val in p['recipe']['genome'].items() if child['genome'][g]!=val]
            if changed!=[v['gene']] or child['family']!=p['family'] or v['numeric_parameters_sha256']!=_digest(p['params']):
                raise ValueError('Not a single typed structural transformation')
            pending=copy.deepcopy(v)
            if v['diagnostic_context_sha256'] not in diagnoses:
                # Historical journal has no separate diagnosis-complete record.
                # This proposal proves one consumption; the timestamp is its
                # recorded upper bound, not an invented diagnosis finish time.
                emit('DIAGNOSIS_COMPLETED',r,parent=current,evidence=ctx['evidence'],
                     context=v['diagnostic_context_sha256'],timestamp_semantics='PROPOSAL_BOUND_UPPER_BOUND')
            emit('MUTATION_PROPOSED',r,parent=current,program=child,gene=v['gene'])
        elif op=='search_budget_stopped':
            if (mode!='PUBLIC_USER' or v.get('retained') not in forms
                    or v.get('stage') not in ('FAMILY','EVOLUTION','NUMERIC')):
                raise ValueError('Unverified typed public budget stop')
            if current is None:current=v['retained']
            if v['retained']!=current:raise ValueError('Budget stop lost the verified current anchor')
            budget_stop=copy.deepcopy(v)
            emit('SEARCH_STOPPED_BUDGET',r,parent=current,details=budget_stop,status='STOPPED_BUDGET')
        elif op=='evolution_interrupted':
            if (pending is None or budget_stop is None or budget_stop['stage']!='EVOLUTION'
                    or v.get('reason')!='TYPED_SEARCH_BUDGET'
                    or v.get('index')!=pending['index'] or v.get('parent')!=current
                    or v.get('retained')!=current or v.get('decision')!='RESTORE_VERIFIED_PARENT'):
                raise ValueError('Invalid incomplete mutation recovery')
            interrupted_steps.append({'index':pending['index'],'parent':current,
                                      'status':'INTERRUPTED_NOT_ADOPTED','retained':current})
            emit('MUTATION_INTERRUPTED',r,parent=current,status='INTERRUPTED_NOT_ADOPTED')
            pending=None
        elif op=='numeric_budget_skipped':
            if frozen is None or budget_stop is None or v.get('retained')!=current:
                raise ValueError('Unverified numerical budget skip')
            emit('NUMERIC_TUNING_SKIPPED',r,parent=current,reason='SEARCH_BUDGET')
        elif op=='evolution_selection' and v['status']=='COMPLETED':
            if pending is None or v['parent']!=current or v['candidate'] not in forms:
                raise ValueError('Child without a verified parent')
            child=checked(v['candidate']);parent=forms[current];cp=child['packet'];pp=parent['packet']
            if (child['program']!=pending['child_program'] or cp['params']!=pp['params']
                    or cp['learning_identity']!=pp['learning_identity']
                    or cp['recipe']['classes']!=pp['recipe']['classes']
                    or cp['recipe']['encoder']!=pp['recipe']['encoder']):
                raise ValueError('Executed child differs from nominated program/data/numbers')
            decision='ACCEPT' if child['SEARCH_primary']<parent['SEARCH_primary'] else 'ROLLBACK'
            retained=v['candidate'] if decision=='ACCEPT' else current
            if v['decision']!=decision or v['retained']!=retained:
                raise ValueError('False acceptance/rejection or retained anchor')
            emit('MUTATED_PROGRAM_EXECUTED',r,parent=current,packet=cp['id'],program=child['program'])
            emit('CANDIDATE_EVALUATED',r,parent=current,SEARCH_parent=parent['SEARCH_primary'],
                 SEARCH_candidate=child['SEARCH_primary'])
            emit('MUTATION_ACCEPTED' if decision=='ACCEPT' else 'MUTATION_REJECTED',r,parent=current,child=cp['id'])
            if decision=='ROLLBACK':emit('ROLLBACK_COMPLETED',r,parent=current,retained=retained)
            steps.append(dict(index=pending['index'],parent=current,child=cp['id'],
                parent_program=parent['program'],child_program=child['program'],decision=decision,
                retained=retained,SEARCH_parent=parent['SEARCH_primary'],SEARCH_candidate=child['SEARCH_primary'],
                SEARCH_retained=forms[retained]['SEARCH_primary'],execution=pending['execution']))
            current=retained;pending=None
        elif op=='evolution_complete':
            if current is None:current=v['retained']
            if current not in forms or v['retained']!=current or v['program']!=forms[current]['program'] or pending:
                raise ValueError('Unverifiable terminal program')
        elif op=='structure_frozen' and 'packet' in v:
            p=v['packet']
            if current is None:current=p['id']
            if current not in forms or p!=packets[current] or pending:raise ValueError('Incompatible frozen anchor')
            if frozen is not None and frozen!=p:raise ValueError('Changed frozen program')
            frozen=copy.deepcopy(p);emit('PROGRAM_FROZEN',r,parent=current,program=program(p),params=p['params'])
            if v.get('numeric_budget')==0:emit('NUMERIC_TUNING_SKIPPED',r,parent=current,reason='ZERO_ADMITTED_NUMERIC_BUDGET')
        elif op=='numeric_selection' and v.get('status')=='COMPLETED':
            if frozen is None or v['retained'] not in forms:raise ValueError('Unverified numeric selection')
            chosen=checked(v['retained'])
            if (chosen['program']!=program(frozen)
                    or chosen['packet']['learning_identity']!=frozen['learning_identity']
                    or chosen['SEARCH_primary']>forms[current]['SEARCH_primary']):
                raise ValueError('Numeric phase changed structure/data or lost verified anchor')
            current=v['retained']
    if (frozen is None or current!=final_packet['id'] or final_packet!=packets[current]
            or program(final_packet)!=program(frozen)):
        raise ValueError('Final model is not the frozen verified anchor')
    # Reopen selected weights again after traversal; verifier must bind payload.
    final=checked(current)
    stop=next((r['value']['stop'] for r in reversed(rows) if r['event']=='ORCHESTRATION'
               and r['value'].get('operation')=='evolution_complete'),'UNKNOWN')
    counts={'physical_fits_started':len(starts),'physical_fits_completed':len(ends),
        'incomplete_native_fits':len(starts.keys()-ends.keys()),
        'scalar_trees_started':sum(r['scalar_trees'] for r in starts.values()),
        'OOF_components':sum(p['components'] for p in contexts.values()),
        'OOF_physical_fits_completed':sum(r['phase']=='OOF' for r in ends.values()),
        'OOF_contexts_acquired':len(contexts),
        'OOF_contexts_reused':sum(r['event']=='OOF_CONTEXT_REUSED' for r in rows),
        'explicit_diagnosis_completions':len(diagnoses),
        'cache_hits':sum(r['event']=='CACHE_HIT' for r in rows),
        'real_trials_started':sum(r['real'] for r in trials.values()),
        'real_trials_completed':sum(r['real'] and r['state']=='COMPLETE' for r in trial_ends.values()),
        'simulated_trials_started':sum(not r['real'] for r in trials.values())}
    if mode=='PUBLIC_USER':
        counts['trial_reservations']=len(trials)
        counts['real_trials_created']=sum(r['real'] for r in created_trials.values())
        counts['simulated_trials_created']=sum(not r['real'] for r in created_trials.values())
    hits={r['key'] for r in rows if r['event']=='CACHE_HIT'}
    fixture_keys={r['key'] for r in rows if r['event']=='FIXTURE_COMPLETED'}
    unassigned=(set(ends)|hits|fixture_keys)-artifact_keys
    interrupted=set(starts)-set(ends);unadopted={}
    if unassigned or interrupted:
        if mode!='PUBLIC_USER' or budget_stop is None or not callable(budget_recovery):
            raise ValueError('Physical/cache/fixture accounting does not match verified artifacts')
        unadopted=budget_recovery(unassigned|interrupted)
        if type(unadopted) is not dict or set(unadopted)!=(unassigned|interrupted):
            raise ValueError('Every unadopted component needs explicit artifact/intent verification')
        for key,p in unadopted.items():
            if (p.get('key')!=key or type(p.get('committed')) is not bool
                    or p.get('used_in_final_model') is not False
                    or not isinstance(p.get('intent_sha256'),str) or len(p['intent_sha256'])!=64
                    or p['committed'] and not isinstance(p.get('artifact'),dict)):
                raise ValueError('Invalid unadopted component evidence')
            p['native_completed']=key in ends
            p['status']='COMMITTED_NOT_ADOPTED' if p['committed'] else 'NATIVE_COMPLETE_NOT_COMMITTED' if key in ends else 'INTERRUPTED_NOT_ADOPTED'
    if artifact_keys-(set(ends)|hits|fixture_keys):
        raise ValueError('Reported component has no physical/cache/fixture evidence')
    def interval_union(keys):
        intervals=sorted((starts[k]['UTC_unix'],ends[k]['UTC_unix']) for k in keys if k in starts and k in ends)
        total=0.;last=None
        for a,b in intervals:
            if b<a:raise ValueError('Native completion precedes start')
            total+=max(0.,b-max(a,last if last is not None else a));last=max(b,last if last is not None else b)
        return total
    for f in forms.values():
        f['native_logged_wall_union_seconds']=interval_union(f['component_keys'])
        f['native_physical_fits']=sum(k in ends for k in f['component_keys'])
        f['cost_scope']='NATIVE_EVENT_WINDOWS_ONLY_NOT_COMPLETE_SEARCH_OR_BENCHMARK'
    # checked() reopens final weights; annotate its same scoped native costs.
    final.update({k:forms[current][k] for k in ('native_logged_wall_union_seconds','native_physical_fits','cost_scope')})
    if mode in ('LIVE','PUBLIC_USER') and counts['OOF_components']!=counts['OOF_physical_fits_completed']:
        # Cache-reused components must be explicitly distinguished by the proof.
        cached=sum(p.get('cache_components',0) for p in contexts.values())
        orphan_oof=sum(k in unadopted and r['phase']=='OOF' for k,r in ends.items())
        if counts['OOF_components']!=counts['OOF_physical_fits_completed']-orphan_oof+cached:
            raise ValueError('OOF component/native accounting contradiction')
    status='MODEL_READY_EXPORT_NOT_VERIFIED'
    if finalization is not None:
        # Verification occurs in the existing postfit safe-loader/V3/standalone
        # path or a read-only replay, never inferred from a report boolean.
        f=finalization(copy.deepcopy(final_packet))
        if f['packet_sha256']!=_digest(final_packet) or f['weights_sha256']!=final_packet['artifact_sha256']:
            raise ValueError('Finalized weights differ from retained program')
        source=next((r for r in reversed(rows) if r['event']=='WORKFLOW_COMPLETED'),None)
        if source is not None:
            if source['model']!=f['model']:raise ValueError('Final artifact differs from workflow')
            emit('MODEL_FINALIZED',source,parent=current,artifacts=f)
            status='COMPLETED_VERIFIED_EXPORT'
    out={'schema':'comp010-verified-evolution-report-v1','status':status,'evidence':mode,
        'journal_head_sha256':previous,'events':events,'forms':forms,'steps':steps,
        'counts':counts,'stop':stop,'final':final,'diagnosis_executed':bool(contexts),
        'diagnostic_utility':'INCONCLUSIVE','performance_qualified':False,
        'OOF_acquisition_method_seconds':sum(p['seconds'] for p in contexts.values()),
        'cost_scope':'INCLUSIVE_OPERATIONAL_METHOD_ONLY_EXCLUDES_OUTER_ADMISSION_AND_REVERIFICATION',
        'CPU':'UNKNOWN','peak_RSS':'UNKNOWN','global_optimum_claim':False}
    if budget_stop is not None:
        out['budget_recovery']={'schema':'product006-verified-budget-recovery-v1',
            'stop':budget_stop,'unadopted_components':unadopted,
            'interrupted_steps':interrupted_steps,'phase_completion':'PARTIAL_BUDGET_VALID_ANCHOR_RETURNED',
            'not_claimed':'No completed mutation/OOF/trial inferred from partial physical work'}
    if claimed is not None:
        # Optional validation is exact; legacy summaries require an explicit
        # discrepancy overlay and must never be silently accepted as verified.
        for key in ('counts','steps','final','diagnosis_executed','journal_head_sha256'):
            if claimed.get(key)!=out[key]:raise ValueError('Stale or contradictory report: '+key)
    return out
