"""Finite installed worker/launcher. No checkout dependency or scheduler.

LIVE requires a separately pinned new owner permit; the current directive only
admits PREFIT and retained REPLAY. Existing ChildGuard/OS leases are reused.
"""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .admission import AdmissionEngine,AdmissionError,read,sha,digest,safe_path,environment
from .runtime_io import configure,artifact_path
from .processes import ChildGuard,exclusive_lock
from .producer import (PROFILE,PHASES,PIPE_CURRENT,Session,PreFitBoundary,prepare,
    bound_reference,validate_permit,atomic_result,live_pipeline,claim_once)


def spawn_guarded(args,*,memory_limits=None,**kwargs):
    """Bind the existing cleanup job before a Windows venv creates descendants.

    Only the primary thread of the child just created by this call is resumed.
    No foreign process or thread is modified. Failure is fail-closed.
    """
    if os.name!='nt':
        child=subprocess.Popen(args,**kwargs);return child,ChildGuard(child,memory_limits=memory_limits)
    import ctypes
    from ctypes import wintypes
    child=subprocess.Popen(args,creationflags=0x00000004,**kwargs) # CREATE_SUSPENDED
    guard=None
    try:
        guard=ChildGuard(child,memory_limits=memory_limits)
        class ThreadEntry(ctypes.Structure):
            _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ThreadID',wintypes.DWORD),('th32OwnerProcessID',wintypes.DWORD),('tpBasePri',wintypes.LONG),('tpDeltaPri',wintypes.LONG),('dwFlags',wintypes.DWORD)]
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];k.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
        k.Thread32First.argtypes=[wintypes.HANDLE,ctypes.POINTER(ThreadEntry)];k.Thread32Next.argtypes=k.Thread32First.argtypes
        k.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenThread.restype=wintypes.HANDLE
        k.ResumeThread.argtypes=[wintypes.HANDLE];k.ResumeThread.restype=wintypes.DWORD
        k.CloseHandle.argtypes=[wintypes.HANDLE]
        snapshot=k.CreateToolhelp32Snapshot(4,0)
        if snapshot==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        tids=[]
        try:
            entry=ThreadEntry();entry.dwSize=ctypes.sizeof(entry);more=k.Thread32First(snapshot,ctypes.byref(entry))
            while more:
                if entry.th32OwnerProcessID==child.pid:tids.append(entry.th32ThreadID)
                more=k.Thread32Next(snapshot,ctypes.byref(entry))
        finally:k.CloseHandle(snapshot)
        if len(tids)!=1:raise RuntimeError('Owned suspended child must have one primary thread')
        thread=k.OpenThread(2,False,tids[0]) # THREAD_SUSPEND_RESUME, owned child only
        if not thread:raise ctypes.WinError(ctypes.get_last_error())
        try:
            if k.ResumeThread(thread)!=1:raise RuntimeError('Unexpected primary-thread suspension count')
        finally:k.CloseHandle(thread)
        return child,guard
    except BaseException:
        if guard:guard.close()
        else:
            if child.poll() is None:child.kill()
            child.wait()
        raise


def ref(root,path):
    p=safe_path(root,path);return {'path':p.relative_to(Path(root).resolve()).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}


def standalone(root,value,check):
    """Inference-only child, no installed library available under -S."""
    root=Path(root);job={'program':value['program'],'expected':value['expected'],'query':value['query']}
    source='''import sys,json,runpy
from pathlib import Path
root=Path(sys.argv[1]);d=json.loads((root/'STANDALONE-INPUT.json').read_text('utf8'))
class Named:
 columns=d['query']['names']
 def to_numpy(self):return d['query']['rows']
p=Path(d['program']['path']);p=p if p.is_absolute() else root/p
query=d['query']['rows'] if d['query']['names'] is None else Named()
ns=runpy.run_path(str(p));actual={'prediction':ns['predict'](query)}
if 'probabilities' in d['expected']:actual['probabilities']=ns['predict_proba'](query)
assert actual==d['expected']
assert not any(n=='tools' or n.startswith('morphtree') or n.startswith('numpy') or n.startswith('optuna') for n in sys.modules)
(root/'STANDALONE-RESULT.json').write_text(json.dumps({'status':'PASS','predictions':actual,'package_imports':0,'native_fits':0,'trials':0}),encoding='utf8')
'''
    # program references are relative to the explicit input root; bind the
    # resolved file into this isolated job, never a checkout path.
    program=safe_path(root,value['program']['path'])
    job['program']={**value['program'],'path':str(program)}
    atomic_result(root,'STANDALONE-INPUT.json',job,check)
    script=safe_path(root,'STANDALONE-CHECK.py')
    from .runtime_io import publish
    publish(script,source,check);check()
    with (root/'STANDALONE.log').open('x',encoding='utf8') as f:
        child,guard=spawn_guarded([sys.executable,'-I','-S','-B',str(script),str(root)],cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
        try:
            while child.poll() is None:
                check();time.sleep(.01)
        finally:guard.close()
    if child.returncode:raise RuntimeError('Standalone failed; original workflow remains failed')
    return read(root/'STANDALONE-RESULT.json')


def verify_commit(root):
    root=Path(root);receipt=read(root/'COMPLETE.json')
    if receipt.get('schema')!='morph031-installed-result-v2' or receipt.get('status') not in ('QUALIFIED_BEFORE_FIT','QUALIFIED_SOFTWARE_PIPELINE','COMPLETED_LIVE'):raise ValueError('Not a complete installed result')
    result=bound_reference(root,receipt['result'])
    for r in receipt['artifacts']:bound_reference(root,r) if r['path'].endswith('.json') else _verify_bytes(root,r)
    if digest(result)!=receipt['result_digest']:raise ValueError('Changed result identity')
    if receipt['status']!='COMPLETED_LIVE' and result['counts']['physical_fits_started']!=0:raise ValueError('Zero-fit result contains training')
    if receipt['status']=='COMPLETED_LIVE' and not (0<result['counts']['physical_fits_completed']<=13 and result['counts']['physical_fits_completed']==result['counts']['physical_fits_started'] and result['counts']['Optuna_trials']==0 and result.get('pipeline_status')=='ALL_PLANNED_STAGES_COMPLETE'):raise ValueError('No completed actual fit')
    return receipt,result


def _verify_bytes(root,r):
    p=safe_path(root,r['path'])
    if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Changed completed artifact')


def commit(root,status,result,check):
    root=Path(root);check()
    if (root/'COMPLETE.json').exists():raise FileExistsError('Never replace completed result')
    # Even if interrupted after RESULT publication, no COMPLETE means no winner.
    rr=atomic_result(root,'RESULT.json',result,check)
    artifacts=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in ('COMPLETE.json','EXIT.json','WORKER.log') and not p.name.endswith('.pending') and 'leases' not in p.parts and p.name not in ('ADMISSION.jsonl','WRITER.lock','.lock'):
            artifacts.append(ref(root,p))
    done={'schema':'morph031-installed-result-v2','status':status,'result':rr,'result_digest':digest(result),'artifacts':artifacts,'native_counts':result['counts'],'training_claim':status=='COMPLETED_LIVE'}
    atomic_result(root,'COMPLETE.json',done,check)
    verify_commit(root)
    return done


def _worker(root,policy_sha):
    root=Path(root).resolve();configure(root.parent if root.parent.name=='runs' else root)
    policy=read(root/'RUN-POLICY.json',expected_sha256=policy_sha)
    # Bootstrap only: no data/model/Optuna work before actual pipe release.
    atomic_result(root,'HELLO.json',{'pid':os.getpid(),'ppid':os.getppid(),'policy_sha256':policy_sha,'environment':environment()},lambda:None)
    if sys.stdin.buffer.read(1)!=b'1':raise PermissionError('Missing pre-operation pipe admission')
    configure(Path(policy['input_root']))
    # Cooperative audit in the real child: no repository lookup is needed to
    # deny every read outside stdlib/installation/explicit supplied inputs.
    allowed=[Path(policy['input_root']).resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path(policy['authority_root']).resolve()]
    def audit(event,args):
        if event=='import' and (args[0]=='tools' or args[0].startswith('tools.')):raise ImportError('Developer namespace forbidden in installed worker')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            try:p=Path(os.fsdecode(args[0])).resolve()
            except (ValueError,OSError):return
            if not any(p.is_relative_to(a) for a in allowed):raise PermissionError('Outside installed/runtime/explicit-input scope')
    sys.addaudithook(audit)
    engine=AdmissionEngine(package=Path(__file__).resolve().parents[1],qualification_path=policy['qualification_path'],qualification_sha256=policy['qualification_sha256'],workdir=root)
    original_entries=[]
    if policy['mode']!='LIVE':
        # Independent software sentinel: a counter alone cannot prove that the
        # original native function was never entered during a zero-fit test.
        native_file=str(Path(__file__).resolve().parents[1]/'model.py')
        optuna_file=str(Path(__file__).resolve().parents[2]/'optuna/study/study.py')
        def no_native_profile(frame,event,arg):
            if event=='call' and frame.f_code.co_filename==optuna_file and frame.f_code.co_name=='ask':raise AssertionError('Actual Optuna ask forbidden in zero-fit worker')
            if event=='call' and frame.f_code.co_filename==native_file and frame.f_code.co_name=='fit':
                original_entries.append(frame.f_code.co_name)
                raise AssertionError('Original native fit forbidden in zero-fit worker')
        sys.setprofile(no_native_profile)
    if policy['environment']!=environment() or policy['worker_sha256']!=sha(__file__):raise ValueError('Actual installed worker/environment changed')
    plan=bound_reference(engine.q['input_root'],policy['plan'])
    owner=bound_reference(engine.q['input_root'],policy['owner_permit']) if policy['owner_permit'] else None
    token=PIPE_CURRENT.set(digest(policy));session=None
    try:
        session=Session(engine,plan,policy,root,owner_permit=owner,owner_text_sha=policy['owner_text_sha256'])
        if policy['scenario']=='before':
            session.prepared={'task':'binary'}
            raise ValueError('SOFTWARE_BEFORE_CALLBACK_FAULT')
        try:__import__('tools.workspace_paths')
        except ImportError:isolation='CHECKOUT_IMPORT_REFUSED'
        else:raise AssertionError('Developer tools unexpectedly importable')
        if policy['mode']=='FIXTURE':
            value=live_pipeline(session)
            # Generated export locators are input-root relative.
            value['program']={**value['program'],'path':str(safe_path(engine.q['input_root'],value['program']['path']))}
            value.setdefault('standalone',{'status':'ALREADY_VERIFIED_PER_WORKFLOW'})
            status='QUALIFIED_SOFTWARE_PIPELINE'
        elif policy['mode']=='PREFIT':
            try:session.execute()
            except PreFitBoundary as exc:
                atomic_result(root,'PREFIT-READY.json',{'status':'BEFORE_ORIGINAL_NATIVE_FIT','key':session.prepared['key'],'counts':session.counters()},session.check)
                scenario=policy['scenario']
                if scenario=='journal':
                    with (session.store.root/'EVENTS.jsonl').open('ab') as f:f.write(b'{')
                    raise ValueError('SOFTWARE_JOURNAL_TAIL_FAULT')
                if scenario=='crash':os._exit(17)
                if scenario=='wait':
                    while True:session.check();time.sleep(.01)
                if scenario=='filesystem':
                    # Intentional conflict at the atomic result destination.
                    (root/'RESULT.json.pending').write_text('partial fixture',encoding='utf8')
                value={'kind':'ZERO_FIT_PREFLIGHT','reason':str(exc),'counts':session.counters(),'native_request':session.prepared['request'],'native_key':session.prepared['key'],'training_executed':False}
                status='QUALIFIED_BEFORE_FIT'
            else:raise AssertionError('Qualification reached an original native fit')
        elif policy['mode']=='LIVE':
            value=live_pipeline(session)
            # Native result paths use the configured input-root locator.
            value['program']={**value['program'],'path':str(safe_path(engine.q['input_root'],value['program']['path']))}
            value.setdefault('standalone',{'status':'ALREADY_VERIFIED_PER_WORKFLOW'});status='COMPLETED_LIVE'
        else:raise ValueError('Unknown worker mode')
        session.check();value.update(mode=policy['mode'],plan_id=plan['id'],runtime=engine.identity,actual_worker_pid=os.getpid(),isolation=isolation,original_native_fit_entries=len(original_entries),origins={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items() if (n=='morphtree' or n.startswith('morphtree.')) and getattr(m,'__file__',None)})
        commit(root,status,value,session.check)
    except BaseException as exc:
        try:failure_counts=session.counters() if session else {'physical_fits_started':0,'physical_fits_completed':0,'Optuna_trials':0}
        except Exception as accounting_error:failure_counts={'status':'UNKNOWN','reason':type(accounting_error).__name__+': '+str(accounting_error),'journal_requires_reconciliation':True}
        if session is not None:engine._event('WORKFLOW_INCOMPLETE',session.prepared.get('key','UNPREPARED') if session.prepared else 'UNPREPARED',error=type(exc).__name__,message=str(exc),counts=failure_counts)
        if not (root/'WORKER-ERROR.json').exists():atomic_result(root,'WORKER-ERROR.json',{'status':'FAIL_OR_INTERRUPTED','error':type(exc).__name__,'message':str(exc),'counts':failure_counts},lambda:None)
        raise
    finally:PIPE_CURRENT.reset(token)


def activate(*,qualification_path,qualification_sha256,plan_ref,root,owner_text,expected_owner_text_sha256):
    """Explicit owner activation, never called by zero-fit qualification.

    Freezes the single deadline before creating the detached permit. The
    expected message SHA is supplied by the trusted authorized controller.
    """
    begin_ns=time.monotonic_ns();epoch=time.time()
    if not expected_owner_text_sha256:raise PermissionError('New explicit owner consent is absent')
    q=read(qualification_path,expected_sha256=qualification_sha256);root=safe_path(q['input_root'],root);configure(q['input_root'])
    if not root.is_dir():raise ValueError('Existing owner-scoped directory required')
    with exclusive_lock(root/'activation-lease'):
        if any((root/n).exists() for n in ('ACTIVATION.json','ATTEMPT.json','RUN-POLICY.json','STOP')):raise RuntimeError('Activation consumed; cannot renew')
        owner=bound_reference(q['input_root'],owner_text)
        if owner_text['sha256']!=expected_owner_text_sha256 or owner.get('authority')!='EXPLICIT_HUMAN_MESSAGE' or owner.get('campaign')!='COMP009-LIVE-001' or owner.get('approved_profile')!=PROFILE:raise PermissionError('Not this new exact owner authorization')
        e=AdmissionEngine(package=Path(__file__).resolve().parents[1],qualification_path=qualification_path,qualification_sha256=qualification_sha256,workdir=root)
        p=bound_reference(q['input_root'],plan_ref);a=prepare(e,p)
        if owner.get('approved_runtime')!=e.identity or owner.get('approved_protocol')!=p['protocol'] or owner.get('authority_root')!=q['authority_root']:raise PermissionError('Owner grant does not bind this exact runtime/protocol/authority scope')
        if a['task']!='multiphase':raise PermissionError('Only exact finite multiphase pilot')
        policy={'schema':'morph031-finite-installed-worker-v2','authority_root':q['authority_root'],'mode':'LIVE','scenario':'normal','input_root':q['input_root'],'qualification_path':str(Path(qualification_path).resolve()),'qualification_sha256':qualification_sha256,'plan':plan_ref,'environment':environment(),'worker_sha256':sha(__file__),
            'created_ns':begin_ns,'deadline_ns':min(begin_ns+int(PROFILE['worker_seconds']*1e9),begin_ns+int(max(0,q['deadline_utc_epoch']-epoch)*1e9)),'deadline_utc':min(epoch+PROFILE['worker_seconds'],q['deadline_utc_epoch'])}
        # INT010 is zero-fit and cannot consume or create a LIVE authority.
        if p['protocol'].get('id')!='COMP-009-LIVE-001' or not e.q.get('training_protocol'):raise PermissionError('Separate frozen COMP003LIVE required before activation')
        physical=bound_reference(q['input_root'],e.q['training_protocol'])
        if physical.get('id')!='COMP-009-LIVE-001' or not physical.get('frozen_before_execution') or physical.get('planned_profile')!=PROFILE:raise PermissionError('Different physical protocol/profile')
        claim_once(q['authority_root'],expected_owner_text_sha256,{'owner_text_sha256':expected_owner_text_sha256,'native_key':a['key'],'output_root':str(root),'policy_sha256':digest(policy)})
        ar=atomic_result(root,'ACTIVATION.json',policy,lambda:None)
        permit={'schema':'morph031-owner-fit-permit-v1','scope':'EXPLICIT_OWNER_APPROVED_COMP009_EVOLUTION','owner_text':owner_text,'runtime':e.identity,'protocol':p['protocol'],'plan_id':p['id'],'native_key':a['key'],'profile':PROFILE,'output_root':str(root),'phases':list(PHASES),'deadline_utc':policy['deadline_utc'],'policy_sha256':digest(policy)}
        pr=atomic_result(root,'OWNER-PERMIT.json',permit,lambda:None)
        return {'activation_ref':ref(q['input_root'],root/ar['path']),'owner_permit':ref(q['input_root'],root/pr['path']),'owner_text_sha256':expected_owner_text_sha256}


def launch(*,qualification_path,qualification_sha256,plan_ref,root,mode='PREFIT',seconds=20,scenario='normal',control=None,owner_permit=None,owner_text_sha256=None,activation_ref=None):
    """One bounded actual child. No retries, resume, cached fitting or scheduler."""
    begin_ns=time.monotonic_ns();begin_epoch=time.time();root=Path(root).resolve()
    if mode not in ('PREFIT','FIXTURE','LIVE') or type(seconds) not in (int,float) or not math_finite(seconds) or not 0<seconds<=(PROFILE['worker_seconds'] if mode=='LIVE' else 540):raise ValueError('Finite exact worker mode/time')
    if scenario not in ('normal','wait','crash','filesystem','journal','before','oof') or control not in (None,'stop','terminate'):raise ValueError('Known qualification fault only')
    q=read(qualification_path,expected_sha256=qualification_sha256);safe_path(q['input_root'],root);configure(q['input_root'])
    if not root.is_dir():raise ValueError('Prevalidated existing run directory required')
    with exclusive_lock(root/'parent-lease'):
        if (root/'ATTEMPT.json').exists() or (root/'RUN-POLICY.json').exists() or (root/'STOP').exists():raise RuntimeError('Run already consumed/stopped; no retry or deadline renewal')
        e=AdmissionEngine(package=Path(__file__).resolve().parents[1],qualification_path=qualification_path,qualification_sha256=qualification_sha256,workdir=root)
        plan=bound_reference(q['input_root'],plan_ref);prepared=prepare(e,plan)
        policy={'schema':'morph031-finite-installed-worker-v2','authority_root':q['authority_root'],'mode':mode,'scenario':scenario,'input_root':q['input_root'],'qualification_path':str(Path(qualification_path).resolve()),'qualification_sha256':qualification_sha256,'plan':plan_ref,'environment':environment(),'worker_sha256':sha(__file__),
            'created_ns':begin_ns,'deadline_ns':min(begin_ns+int(seconds*1e9),begin_ns+int(max(0,q['deadline_utc_epoch']-begin_epoch)*1e9)),'deadline_utc':min(begin_epoch+seconds,q['deadline_utc_epoch']),'owner_permit':owner_permit,'owner_text_sha256':owner_text_sha256}
        if mode=='LIVE':
            if scenario!='normal' or control is not None:raise ValueError('No injected faults in a physical pilot')
            if not owner_permit or not owner_text_sha256 or not activation_ref:raise PermissionError('Separate new owner activation/permit absent')
            activation=bound_reference(q['input_root'],activation_ref)
            for k in ('schema','mode','scenario','input_root','qualification_path','qualification_sha256','plan','environment','worker_sha256'):
                if activation[k]!=policy[k]:raise ValueError('Different activation source/environment/request')
            policy={**activation,'owner_permit':owner_permit,'owner_text_sha256':owner_text_sha256}
            begin_ns=policy['created_ns']
            permit=bound_reference(q['input_root'],owner_permit) if owner_permit else None
            validate_permit(permit,expected_owner_text_sha=owner_text_sha256,engine=e,prepared=prepared,policy=policy,workdir=root)
        atomic_result(root,'RUN-POLICY.json',policy,lambda:None)
        policy_file=root/'RUN-POLICY.json';pin=sha(policy_file)
        atomic_result(root,'ATTEMPT.json',{'status':'INTENT_BEFORE_PROCESS','plan_id':plan['id'],'mode':mode,'native_calls':0,'trial_calls':0,'policy_sha256':pin},lambda:None)
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        reason=None;hello=None
        with (root/'WORKER.log').open('x',encoding='utf8') as log:
            child,guard=spawn_guarded([sys.executable,'-I','-B','-m','morphtree._experimental.worker',str(root),pin],cwd=root,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt')
            try:
                while child.poll() is None and not (root/'HELLO.json').exists():
                    if time.monotonic_ns()>=policy['deadline_ns']:raise TimeoutError('Bootstrap deadline')
                    time.sleep(.01)
                hello=read(root/'HELLO.json')
                if hello['policy_sha256']!=pin or hello['environment']!=environment() or hello['pid']!=child.pid and hello['ppid']!=child.pid:raise ValueError('Unknown bootstrap process identity')
                guard.include_pid(hello['pid'])
                child.stdin.write(b'1');child.stdin.close()
                while child.poll() is None:
                    ready=(root/'PREFIT-READY.json').exists()
                    if ready and control=='stop':(root/'STOP').write_text('owner-controlled zero-fit test STOP',encoding='utf8');reason='STOP';break
                    if ready and control=='terminate':reason='CONTROLLED_PROCESS_INTERRUPTION';break
                    if time.monotonic_ns()>=policy['deadline_ns'] or time.time()>=policy['deadline_utc']:reason='DEADLINE';break
                    if (root/'STOP').exists():reason='STOP';break
                    time.sleep(.01)
            except Exception as exc:reason='BOOTSTRAP_OR_PROCESS:'+type(exc).__name__+': '+str(exc)
            finally:guard.close()
        rc=child.wait(timeout=5)
        completed=False
        if rc==0 and reason is None:
            try:receipt,value=verify_commit(root);completed=True
            except Exception as exc:reason='RESULT_VERIFICATION:'+type(exc).__name__+': '+str(exc)
        exit={'status':'SOFTWARE_QUALIFIED' if completed and mode!='LIVE' else 'COMPLETED_LIVE' if completed else 'INCOMPLETE','returncode':rc,'reason':reason,'mode':mode,'actual_worker_pid':hello['pid'] if hello else None,'launcher_pid':child.pid,'guard_closed':True,'native_training_claim':completed and mode=='LIVE','operational_seconds_with_cleanup':time.monotonic_ns()/1e9-begin_ns/1e9,'retry':False}
        atomic_result(root,'EXIT.json',exit,lambda:None)
        if not (root/'STOP').exists():(root/'STOP').write_text('terminal finite run',encoding='utf8')
        return exit


def math_finite(value):
    import math
    return math.isfinite(value)


PUBLIC_PROFILE={'calls':51,'scalar_trees':90,'trials':6,'global_seconds':600,
    'worker_seconds':540,'call_seconds':150,'component_seconds':60,'workers':1,
    'artifact_bytes':128*1024**2,'process_commit_bytes':768*1024**2,'job_commit_bytes':1024**3,
    'retries':0,'refits':0,'calibrators':0,'GUARD':0,'HTTP':0}
PUBLIC_CAMPAIGN='PRODUCT007-DEFAULT-LIVE-001'


def public_artifact_bytes(root):
    """Checked size,not a disk quota;renamed serializer temporaries may vanish."""
    import stat
    total=0
    for p in Path(root).rglob('*'):
        try:s=p.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400:
            raise ValueError('Public campaign artifact reparse/symlink escape')
        if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total


def public_resource_availability(root):
    """One safety snapshot,not an exclusive host or performance admission."""
    import ctypes,shutil
    if os.name!='nt':raise RuntimeError('Owned private-commit resource profile qualified on Windows only')
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_uint64) for n in ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','extended_virtual')]
    value=MemoryStatus();value.length=ctypes.sizeof(value)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatus)]
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(value)):raise OSError('Cannot read resource availability')
    disk=shutil.disk_usage(root)
    if disk.free<4*PUBLIC_PROFILE['artifact_bytes'] or min(value.available_physical,value.available_commit)<PUBLIC_PROFILE['job_commit_bytes']:
        raise RuntimeError('Insufficient point-in-time disk or memory availability;no retry')
    return {'available_physical_bytes':int(value.available_physical),'available_commit_bytes':int(value.available_commit),
        'free_disk_bytes':disk.free,'exclusive_host':False,'foreign_processes_inspected_or_modified':False,
        'performance_admission':False,'scope':'Point-in-time safety observation;resources are not reserved against unrelated workloads'}


def _public_protocol(path,pin,*,verify_inputs=True):
    """Exact owned qualification profile,not an ordinary-user requirement."""
    import math
    from morphtree import api
    from . import admission as A
    path=Path(path).resolve();p=read(path,expected_sha256=pin)
    keys={'schema','id','profile','runtime','wheel','optional','sources','prefit_models','authority_root','not_after'}
    if set(p)!=keys or p['schema']!='product007-public-protocol-v1' or p['id']!=PUBLIC_CAMPAIGN or p['profile']!=PUBLIC_PROFILE:
        raise ValueError('Exact public qualification protocol/profile required')
    if type(p['not_after']) not in (int,float) or not math.isfinite(p['not_after']) or time.time()>=p['not_after']:
        raise PermissionError('Public qualification protocol expired')
    if p['authority_root']!='authorities':raise ValueError('Single canonical authority root required')
    package=Path(api.__file__).parent
    q={'usage':'ORDINARY_USER_NOT_SCIENTIFIC',**p['runtime']}
    identity=A.installed_identity(package,q)
    if api.PUBLIC_USER_RUNTIME!='1.0.0rc4' or identity['version']!=api.PUBLIC_USER_RUNTIME:
        raise ValueError('Exact qualified product004 runtime required')
    # A wheel is binary: bind it directly, never parse it as JSON.
    wheel=p['wheel'];wf=safe_path(path.parent,wheel['path'])
    if set(wheel)!={'path','sha256','bytes'} or wf.stat().st_size!=wheel['bytes'] or sha(wf)!=wheel['sha256']:
        raise ValueError('Qualified wheel bytes changed')
    if set(p['sources'])!={'binary','multiclass','regression'} or set(p['prefit_models'])!=set(p['sources']):
        raise ValueError('Three exact DEV sources and retained IO models required')
    optional=p['optional']
    if set(optional)!={'version','files'} or optional['version']!='5.0.0' or not 1<=len(optional['files'])<=4096:
        raise ValueError('Exact optional dependency graph required')
    for n,h in optional['files'].items():
        if Path(n).parts[0]=='morphtree' or sha(safe_path(package.parent,n))!=h:raise ValueError('Changed optional dependency')
    if verify_inputs:
        from .target import detect_target,_split
        for task,s in p['sources'].items():
            if set(s)!={'data','exposure','split'} or s['exposure']!='DEV_EXPOSED':raise PermissionError('Only frozen exposed DEV source')
            raw=bound_reference(path.parent,s['data']);actual,y,classes=detect_target(raw['y'],task)
            ti,si=_split(y,task,42)
            if actual!=task or len(raw['X'])!=48 or any(len(row)!=3 for row in raw['X']) or s['split']!={'TRAIN':ti,'SEARCH':si,'GUARD':[]}:
                raise ValueError('Changed DEV rows/task/split')
            model=safe_path(path.parent,p['prefit_models'][task]['path']);r=p['prefit_models'][task]
            if model.stat().st_size!=r['bytes'] or sha(model)!=r['sha256']:raise ValueError('Changed retained IO model')
    return p,identity


class PublicCallBoundary:
    """Existing worker's shared resource/custody boundary;no learning policy."""
    def __init__(self,root,policy,protocol):
        from . import component_cache_v2 as cache
        self.root=Path(root).resolve();self.policy=copy.deepcopy(policy);self.protocol=copy.deepcopy(protocol)
        self.pin=digest(self.policy);self.protocol_pin=digest(protocol);self.closed=False;self.cursor=0
        self.calls=safe_path(self.root,'calls');self.calls.mkdir(exist_ok=True)
        self.sessions=[];self.temporary_roots=[]
        configure(self.root);cache.configure(self.root,Path(__file__).parents[1])
        self.store=cache.MultitaskStore(safe_path(self.root,'global-store'),max_compute_jobs=PUBLIC_PROFILE['calls'],seconds=PUBLIC_PROFILE['worker_seconds'])
        self.check()

    def check(self):
        if self.closed or (self.root/'STOP').exists():raise RuntimeError('Public campaign STOP permanent')
        if time.time()>=self.policy['deadline_utc'] or time.monotonic_ns()>=self.policy['deadline_ns']:
            raise TimeoutError('One immutable global campaign deadline')

    def integrity(self):
        self.check()
        if digest(self.policy)!=self.pin or digest(self.protocol)!=self.protocol_pin or sha(self.root/'PUBLIC-POLICY.json')!=self.policy['policy_file_sha256']:
            raise ValueError('Mutated campaign binding')
        if sha(Path(self.policy['protocol']['path']))!=self.policy['protocol']['sha256']:
            raise ValueError('Mutated campaign protocol')
        if public_artifact_bytes(self.root)>PUBLIC_PROFILE['artifact_bytes']:
            raise RuntimeError('Checked artifact budget exceeded;not a disk quota')

    def _global(self):
        from contextlib import contextmanager
        from .runtime_io import WORKDIR
        @contextmanager
        def scope():
            token=WORKDIR.set(self.root)
            try:yield
            finally:WORKDIR.reset(token)
        return scope()

    def event(self,kind,**fields):
        with self._global():self.store.event(kind,**fields)

    def used(self):
        with self._global():rows=self.store.events()
        intents=[r for r in rows if r['event']=='PUBLIC_LEARNING_INTENT']
        return {'calls':sum(r['operation']=='fit' for r in intents),'scalar_trees':sum(r['trees'] for r in intents),'trials':sum(r['operation']=='trial' for r in intents)}

    def reserve(self,session,*,operation,trees,key=None):
        self.integrity()
        if session not in self.sessions or operation not in ('fit','trial') or type(trees) is not int or trees<0 or operation=='trial' and trees:
            raise PermissionError('Unknown nested operation/session')
        if session.phase not in ('FAMILY','OOF','STRUCTURE','NUMERIC') or operation=='trial' and (session.phase!='NUMERIC' or session.frozen is None):
            raise PermissionError('No numerical reservation before structural freeze')
        with self._global(),exclusive_lock(safe_path(self.root,'public-budget-lock')):
            used=self.used();delta={'calls':int(operation=='fit'),'scalar_trees':trees,'trials':int(operation=='trial')}
            if any(used[k]+v>PUBLIC_PROFILE[k] for k,v in delta.items()):raise RuntimeError('Shared public campaign budget exhausted')
            if key is None and self.policy['mode']=='PREFIT':key=digest({'software_only':used,'operation':operation,'call':str(session.root)})
            if type(key) is not str or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Canonical nested operation key required')
            self.store.event('PUBLIC_LEARNING_INTENT',operation=operation,key=key,trees=trees,call_root=str(session.root),phase=session.phase,
                evidence='PREFIT_RESERVATION_NOT_LEARNING' if self.policy['mode']=='PREFIT' else 'OWNER_BOUND_LIVE_RESERVATION_NOT_COMPLETION')

    def before_input(self,model):
        self.integrity()
        if self.cursor>=3 or model.task!='auto' or (model.families,model.morphing,model.tuning)!=('auto','auto','optuna') or model.random_state!=42 or model.time_budget!='auto' or model.n_trials!='auto' or model.n_jobs not in ('auto',1) or model.capacity is not None or model._max_mutations!=2:
            raise PermissionError('Exact next public API call required')

    def input_rows(self,rows,y,names,task):
        self.integrity();expected=('binary','multiclass','regression')[self.cursor]
        r=self.protocol['sources'][expected]['data'];raw=bound_reference(Path(self.policy['protocol']['path']).parent,r)
        if task!=expected or rows!=raw['X'] or y!=raw['y'] or names is not None:
            raise ValueError('Public qualification input differs from frozen plain lists')

    def retain_temporary(self,temporary,root):
        self.integrity();root=safe_path(self.calls,root)
        # Qualification-owned custody only:normal user calls keep normal cleanup.
        temporary._finalizer.detach();self.temporary_roots.append(str(root))
        self.event('PUBLIC_CALL_DIRECTORY_RETAINED',path=str(root),call_index=self.cursor)

    def bind_session(self,session):
        self.integrity()
        if str(session.root) not in self.temporary_roots or len(self.sessions)!=self.cursor:
            raise PermissionError('Unexpected concurrent or unretained user Session')
        self.sessions.append(session)
        self.event('PUBLIC_SESSION_BOUND',call_index=self.cursor,session_sha256=session.policy_sha,source_sha256=digest(session.source))

    def output(self,path):
        self.integrity();return safe_path(self.root,path)

    def end_call(self,*,status):
        self.integrity();s=self.sessions[-1]
        if not s._closed:raise RuntimeError('Public call not closed')
        self.event('PUBLIC_CALL_CLOSED',call_index=self.cursor,status=status,counts=s.counters())
        self.cursor+=1

    def close(self,reason):
        if self.closed:return
        try:self.event('PUBLIC_CAMPAIGN_CLOSED',reason=reason,reservations=self.used(),authority_residual=0)
        finally:
            self.closed=True
            if not (self.root/'STOP').exists():(self.root/'STOP').write_text(reason,encoding='utf8')


def run_public_qualification(root,protocol_path,protocol_sha256,*,mode='PREFIT',owner_text=None,expected_owner_sha256=None,scenario='normal'):
    """Single owned worker;no valid LIVE entry without separate fresh owner SHA.

    The expected SHA comes from the trusted human-facing caller,not a permit or
    a model. This cooperative boundary cannot authenticate hostile Python code.
    """
    from .producer import _original_claim_once
    from .runtime_io import WORKDIR
    if mode not in ('PREFIT','LIVE') or scenario not in ('normal','wait','crash') or mode=='LIVE' and scenario!='normal':
        raise ValueError('Known finite qualification mode only')
    if mode=='LIVE' and (not owner_text or not expected_owner_sha256):raise PermissionError('Separate fresh human authorization absent')
    entry=time.monotonic_ns();epoch=time.time();protocol_path=Path(protocol_path).resolve()
    # Reject before source rows are materialized. A valid fresh human token is
    # consumed before full preflight;preflight failure cannot recycle it.
    root=safe_path(protocol_path.parent,root)
    authority=safe_path(protocol_path.parent,'authorities');authority.mkdir(exist_ok=True)
    if mode=='LIVE':
        token=expected_owner_sha256
    else:token=digest({'software_only':str(root),'protocol':protocol_sha256})
    previous=WORKDIR.get();configure(authority)
    try:claim=_original_claim_once(authority,token,{'schema':'product007-public-consumption-v1','mode':mode,'campaign':PUBLIC_CAMPAIGN,'protocol_sha256':protocol_sha256,'output_root':str(root),'owner_sha256':expected_owner_sha256 if mode=='LIVE' else None})
    finally:
        if previous is not None:configure(previous)
    if root.exists():
        configure(authority)
        atomic_result(authority/'.native-authorities'/token,'CLOSED.json',{'status':'INCOMPLETE','reason':'OUTPUT_ALREADY_EXISTS','authority_residual':0},lambda:None)
        raise FileExistsError('Never replay qualification output;binding now consumed')
    root.mkdir();configure(root);child=None;guard=None;reason=None;statistics=None;hello=None
    try:
        owner_record=None
        if mode=='LIVE':
            owner=safe_path(protocol_path.parent,owner_text)
            owner_record=read(owner,expected_sha256=expected_owner_sha256)
            required={'schema','authority','campaign','protocol_sha256','runtime_manifest_sha256','profile','message','created_utc_epoch','expires_utc_epoch'}
            if set(owner_record)!=required or owner_record['schema']!='product007-explicit-owner-v1' or owner_record['authority']!='EXPLICIT_HUMAN_MESSAGE' or owner_record['campaign']!=PUBLIC_CAMPAIGN or owner_record['protocol_sha256']!=protocol_sha256 or owner_record['profile']!=PUBLIC_PROFILE or not isinstance(owner_record['message'],str) or not owner_record['message']:
                raise PermissionError('Exact fresh owner scope absent;metadata is not human authorization')
            import math
            if type(owner_record['created_utc_epoch']) not in (int,float) or type(owner_record['expires_utc_epoch']) not in (int,float) or not math.isfinite(owner_record['created_utc_epoch']) or not math.isfinite(owner_record['expires_utc_epoch']) or not owner_record['created_utc_epoch']<=time.time()<owner_record['expires_utc_epoch']:
                raise PermissionError('Owner grant expired or not yet valid')
        p,identity=_public_protocol(protocol_path,protocol_sha256)
        resources=public_resource_availability(root)
        if mode=='LIVE' and owner_record['runtime_manifest_sha256']!=p['runtime']['manifest_sha256']:
            raise PermissionError('Owner grant binds another installed runtime')
        remaining=min(PUBLIC_PROFILE['worker_seconds'],p['not_after']-epoch,
            PUBLIC_PROFILE['global_seconds'] if owner_record is None else owner_record['expires_utc_epoch']-epoch)
        policy={'schema':'product004-owned-public-policy-v1','mode':mode,'scenario':scenario,'protocol':{'path':str(protocol_path),'sha256':protocol_sha256},
            'profile':PUBLIC_PROFILE,'runtime':identity,'resources_before_worker':resources,'created_ns':entry,'deadline_ns':entry+int(remaining*1e9),
            'deadline_utc':min(epoch+PUBLIC_PROFILE['global_seconds'],p['not_after'],float('inf') if owner_record is None else owner_record['expires_utc_epoch']),'owner_sha256':expected_owner_sha256 if mode=='LIVE' else None,
            'owner_text':None if mode=='PREFIT' else {'path':str(owner),'sha256':expected_owner_sha256,'bytes':owner.stat().st_size},
            'claim':{'path':str(authority/'.native-authorities'/token/'CLAIM.json'),'sha256':sha(authority/'.native-authorities'/token/'CLAIM.json')}}
        atomic_result(root,'PUBLIC-POLICY.json',policy,lambda:None);pin=sha(root/'PUBLIC-POLICY.json')
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (root/'WORKER.log').open('x',encoding='utf8') as log:
            child,guard=spawn_guarded([sys.executable,'-I','-B','-m','morphtree._experimental.worker','public-child',str(root),pin],cwd=root,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt',
                memory_limits={k:PUBLIC_PROFILE[k] for k in ('process_commit_bytes','job_commit_bytes')})
            while not (root/'PUBLIC-HELLO.json').exists():
                if child.poll() is not None:raise RuntimeError('Public bootstrap failed')
                if time.monotonic_ns()>=policy['deadline_ns']:raise TimeoutError('Bootstrap deadline')
                time.sleep(.02)
            hello=read(root/'PUBLIC-HELLO.json')
            if hello['policy_sha256']!=pin or hello['pid']!=child.pid and hello['ppid']!=child.pid:raise ValueError('Unknown actual worker')
            guard.include_pid(hello['pid']);statistics=guard.statistics()
            child.stdin.write(b'1');child.stdin.close()
            while child.poll() is None:
                if (root/'STOP').exists() and not (root/'PUBLIC-RESULT.json').exists():raise RuntimeError('STOP')
                if time.monotonic_ns()>=policy['deadline_ns'] or time.time()>=policy['deadline_utc']:raise TimeoutError('Global deadline')
                if public_artifact_bytes(root)>PUBLIC_PROFILE['artifact_bytes']:raise RuntimeError('Checked artifact cap')
                statistics=guard.statistics()
                if statistics['peak_over_job_policy_bytes']:
                    raise RuntimeError('Observed private commit peak exceeded allocation policy;no absolute peak guarantee')
                time.sleep(.05)
            statistics=guard.statistics()
            if child.returncode:raise RuntimeError('Public worker failed;no retry')
            result=read(root/'PUBLIC-RESULT.json')
            if result['status']!='COMPLETE' or result['mode']!=mode or len(result['cases'])!=3:
                raise ValueError('Incomplete public result')
            from types import SimpleNamespace
            from . import component_cache_v2 as cache
            rows=cache.MultitaskStore.events(SimpleNamespace(root=root/'global-store'))
            starts=trees=asked=0
            for row in rows:
                if row['event']!='PUBLIC_SESSION_BOUND':continue
                # Directory is present before bind and never cleaned by GC.
                index=row['call_index'];directory=next(r['path'] for r in rows if r['event']=='PUBLIC_CALL_DIRECTORY_RETAINED' and r['call_index']==index)
                folder=safe_path(root,directory)
                if sha(folder/'SESSION.json')!=row['session_sha256']:raise ValueError('Changed retained session')
                events=cache.MultitaskStore.events(SimpleNamespace(root=folder/'store'))
                starts+=sum(e['event']=='NATIVE_FIT_STARTED' for e in events)
                trees+=sum(e.get('scalar_trees',0) for e in events if e['event']=='NATIVE_FIT_STARTED')
                asked+=sum(e['event']=='NUMERIC_TRIAL_ASKED' and e.get('real',False) for e in events)
            if mode=='PREFIT' and (starts or trees or asked):raise ValueError('PREFIT performed learning')
            if mode=='LIVE' and (not 0<starts<=PUBLIC_PROFILE['calls'] or not 0<trees<=PUBLIC_PROFILE['scalar_trees'] or not 0<=asked<=PUBLIC_PROFILE['trials']):
                raise ValueError('Incomplete actual three-task learning/trials')
            atomic_result(root,'PUBLIC-ACCOUNTING.json',{'native_fits_started':starts,'scalar_trees_started':trees,'actual_Optuna_trials_created':asked,'reservations':result['global_reservations'],'mode':mode},lambda:None)
    except BaseException as exc:reason=type(exc).__name__+': '+str(exc)
    finally:
        if guard is not None:
            try:statistics=guard.statistics()
            except Exception as exc:
                statistics={'status':'UNKNOWN','reason':type(exc).__name__+': '+str(exc)}
                reason=reason or 'RESOURCE_ACCOUNTING_UNKNOWN'
            finally:guard.close()
        elif child is not None and child.poll() is None:child.kill();child.wait()
        configure(root)
        for call in (root/'calls').glob('morphtree-user-*') if (root/'calls').exists() else []:
            call=safe_path(root,call)
            if call.is_dir() and not (call/'STOP').exists():(call/'STOP').write_text('Owned campaign terminated;no resumed learning',encoding='utf8')
        closure={'schema':'product007-public-closure-v1','status':'PASS_PREFIT' if reason is None and mode=='PREFIT' else 'COMPLETED_LIVE' if reason is None else 'INCOMPLETE',
            'mode':mode,'reason':reason,'returncode':None if child is None else child.returncode,'owned_guard_closed':guard is not None,'resource_observation':statistics,
            'authority_residual':0,'retry':False,'elapsed_operational_seconds':(time.monotonic_ns()-entry)/1e9,'worker':hello}
        atomic_result(root,'PUBLIC-EXIT.json',closure,lambda:None)
        if not (root/'STOP').exists():(root/'STOP').write_text('terminal campaign;no authority remains',encoding='utf8')
        # The canonical claim already makes interruption nonrenewable.Closure
        # copies are additional evidence;they never reset/replace that claim.
        configure(authority)
        atomic_result(authority/'.native-authorities'/token,'CLOSED.json',closure,lambda:None)
        configure(root)
    return closure


def _public_worker(root,pin):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    from morphtree import MorphTree,api
    from morphtree.booster import BinaryHistogramBooster
    from morphtree.tree import HistogramTree
    from . import execution_v2 as E
    from .postfit import finish
    root=Path(root).resolve();configure(root);policy=read(root/'PUBLIC-POLICY.json',expected_sha256=pin)
    atomic_result(root,'PUBLIC-HELLO.json',{'pid':os.getpid(),'ppid':os.getppid(),'policy_sha256':pin},lambda:None)
    if sys.stdin.read(1)!='1':raise PermissionError('No owned parent GO')
    p,identity=_public_protocol(policy['protocol']['path'],policy['protocol']['sha256'])
    if identity!=policy['runtime'] or policy['profile']!=PUBLIC_PROFILE or policy['mode'] not in ('PREFIT','LIVE'):
        raise ValueError('Different public policy/runtime')
    claim=read(policy['claim']['path'],expected_sha256=policy['claim']['sha256'])
    if claim['mode']!=policy['mode'] or claim['protocol_sha256']!=policy['protocol']['sha256'] or claim['output_root']!=str(root) or claim['owner_sha256']!=policy['owner_sha256']:
        raise PermissionError('Actual consumed claim does not match')
    if policy['mode']=='LIVE':
        if not policy['owner_sha256'] or not policy.get('owner_text') or claim['campaign']!=PUBLIC_CAMPAIGN:
            raise PermissionError('Actual owner binding missing')
        owner=policy['owner_text'];path=safe_path(Path(policy['protocol']['path']).parent,owner['path'])
        if path.stat().st_size!=owner['bytes'] or sha(path)!=owner['sha256'] or owner['sha256']!=policy['owner_sha256']:
            raise PermissionError('Actual owner text changed')
        permission=read(path,expected_sha256=owner['sha256'])
        if permission['authority']!='EXPLICIT_HUMAN_MESSAGE' or permission['campaign']!=PUBLIC_CAMPAIGN or permission['protocol_sha256']!=policy['protocol']['sha256'] or permission['runtime_manifest_sha256']!=p['runtime']['manifest_sha256'] or permission['profile']!=PUBLIC_PROFILE or not permission['created_utc_epoch']<=time.time()<permission['expires_utc_epoch']:
            raise PermissionError('Actual owner scope expired or differs')
    policy['policy_file_sha256']=pin;scope=PublicCallBoundary(root,policy,p);results=[];models=[]
    token=E.PUBLIC_SCOPE.set(scope)
    try:
        with ExitStack() as stack:
            if policy['mode']=='PREFIT':
                import optuna
                def stop(session,model,X,y,**kwargs):
                    session._user_integrity();scope.event('PUBLIC_PREFIT_BOUNDARY',task=session.prepared['task'])
                    if policy['scenario']=='crash':os._exit(17)
                    if policy['scenario']=='wait':
                        while True:scope.check();time.sleep(.02)
                    raise E.PreFitBoundary('STOP_BEFORE_ORIGINAL_NATIVE_FIT')
                stack.enter_context(patch.object(E.Session,'execute_native',stop))
                stack.enter_context(patch.object(BinaryHistogramBooster,'fit',side_effect=AssertionError('NO_NATIVE_FIT_IN_PREFIT')))
                stack.enter_context(patch.object(HistogramTree,'fit',side_effect=AssertionError('NO_TREE_FIT_IN_PREFIT')))
                stack.enter_context(patch.object(optuna,'create_study',side_effect=AssertionError('NO_STUDY_IN_PREFIT')))
            for task in ('binary','multiclass','regression'):
                scope.check();source=p['sources'][task]
                raw=bound_reference(Path(policy['protocol']['path']).parent,source['data'])
                model=MorphTree(random_state=42,verbosity=0);models.append(model)
                try:model.fit(raw['X'],raw['y'])
                except E.PreFitBoundary:
                    if policy['mode']!='PREFIT':raise
                session=scope.sessions[-1]
                output=safe_path(root,'result-'+task);output.mkdir()
                if policy['mode']=='PREFIT':
                    import shutil
                    f=p['prefit_models'][task];src=safe_path(Path(policy['protocol']['path']).parent,f['path']);dest=output/'RETAINED-INPUT.json'
                    shutil.copyfile(src,dest);model=MorphTree.load(dest,sha256=f['sha256'],workdir=root)
                else:raw={**raw,'names':None} # ordinary fit used plain lists
                configure(root)
                io=finish(SimpleNamespace(check=scope.check,engine=session.engine,source=session.source,root=root,mode='PUBLIC_POSTFIT_IO',counters=session.counters),model,output,raw)
                scope.end_call(status='PREFIT_AND_RETAINED_IO' if policy['mode']=='PREFIT' else 'LIVE_AND_POSTFIT_IO')
                results.append({'task':task,'training':'NOT_EXECUTED' if policy['mode']=='PREFIT' else 'ACTUAL_PUBLIC_API','counts':session.counters(),'postfit':io})
        configure(root)
        atomic_result(root,'PUBLIC-RESULT.json',{'status':'COMPLETE','mode':policy['mode'],'cases':results,'global_reservations':scope.used(),'functional_complete':False,'scientific_adaptive_benefit':'NOT_EVALUATED'},scope.check)
    except BaseException as exc:
        scope.event('PUBLIC_WORKER_ERROR',error=type(exc).__name__,message=str(exc));raise
    finally:
        try:scope.close('terminal public qualification')
        finally:E.PUBLIC_SCOPE.reset(token)


if __name__=='__main__':
    if sys.argv[1]=='public-child':_public_worker(sys.argv[2],sys.argv[3])
    else:_worker(sys.argv[1],sys.argv[2])
