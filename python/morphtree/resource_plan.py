"""Finite workload-based allowances, not dataset size admission thresholds.

Coefficients are conservative engineering heuristics, not a completion-time or
RSS theorem. Explicit user allowances always win. No target values, scores or
EXTERNAL data enter the plan; no samples or evolutionary stages are removed.
"""
import math


def memory_snapshot(host_available):
    """Read-only OS limits, without changing a job/cgroup or foreign process.

    Windows scope is the immediate job plus an optional supervisor-verified
    process ceiling. Numeric context never holds a job handle, which would keep
    a kill-on-last-handle-close cleanup job alive after its owner died.
    Linux scope is the visible cgroup hierarchy and RLIMIT_AS. Unsupported or
    unreadable limits are reported, never claimed as fully qualified coverage.
    """
    import os
    constraints=[];unavailable=[];private=None
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',w.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
        class IO(ctypes.Structure):_fields_=[(n,ctypes.c_uint64) for n in ('ro','wo','oo','rb','wb','ob')]
        class Extended(ctypes.Structure):_fields_=[('basic',Basic),('io',IO),('process_limit',ctypes.c_size_t),('job_limit',ctypes.c_size_t),('process_peak',ctypes.c_size_t),('job_peak',ctypes.c_size_t)]
        class Usage(ctypes.Structure):_fields_=[('current',ctypes.c_uint64),('peak',ctypes.c_uint64)]
        class Process(ctypes.Structure):_fields_=[('size',w.DWORD),('faults',w.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.c_void_p,w.DWORD]
        current_process=Process();current_process.size=ctypes.sizeof(current_process)
        if psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(current_process),current_process.size):private=current_process.private
        else:unavailable.append('Windows_process_commit_usage')
        for handle,scope in [(None,'immediate')]:
            limits=Extended()
            if kernel.QueryInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits),None):
                if limits.basic.flags&0x100:
                    if private is not None:constraints.append(dict(kind='Windows_process_commit',job_scope=scope,limit=limits.process_limit,current=private))
                    else:unavailable.append(scope+'_process_commit_usage')
                if limits.basic.flags&0x200:
                    current=Usage()
                    if kernel.QueryInformationJobObject(handle,28,ctypes.byref(current),ctypes.sizeof(current),None):constraints.append(dict(kind='Windows_job_commit',job_scope=scope,limit=limits.job_limit,current=current.current))
                    else:unavailable.append(scope+'_job_current_usage')
            else:unavailable.append(scope+'_job_unavailable')
        declared=getattr(memory_snapshot,'_supervised_process_limit',None)
        if declared is not None:
            if type(declared) is not int or declared<1:raise ValueError('Positive supervisor-verified process ceiling')
            if private is None:raise MemoryError('Cannot measure process commit under supervisor ceiling')
            constraints.append(dict(kind='Windows_process_commit',job_scope='supervisor_verified_policy',limit=declared,current=private))
        unavailable.append('Other_Windows_ancestor_jobs_not_enumerated')
    elif os.name=='posix':
        from pathlib import Path
        try:
            entries=Path('/proc/self/cgroup').read_text('ascii').splitlines()
            for entry in entries:
                hierarchy,controllers,relative=entry.split(':',2)
                if hierarchy=='0' and not controllers:base=Path('/sys/fs/cgroup');names=('memory.max','memory.current')
                elif 'memory' in controllers.split(','):base=Path('/sys/fs/cgroup/memory');names=('memory.limit_in_bytes','memory.usage_in_bytes')
                else:continue
                current=(base/relative.lstrip('/')).resolve()
                if not current.is_relative_to(base):raise ValueError('Unsafe cgroup path')
                for parent in (current,*current.parents):
                    if not parent.is_relative_to(base):break
                    if not (parent/names[0]).exists():continue
                    maximum=(parent/names[0]).read_text('ascii').strip()
                    if maximum!='max':constraints.append(dict(kind='Linux_visible_cgroup',limit=int(maximum),current=int((parent/names[1]).read_text('ascii').strip())))
        except (OSError,ValueError):unavailable.append('Linux_cgroup_limits')
        try:
            import resource
            limit=resource.getrlimit(resource.RLIMIT_AS)[0]
            if limit!=resource.RLIM_INFINITY:
                status=Path('/proc/self/status').read_text('ascii');virtual=next(int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmSize:'))
                constraints.append(dict(kind='POSIX_address_space',limit=limit,current=virtual))
        except (OSError,ValueError,StopIteration,ImportError):unavailable.append('POSIX_address_space_limit')
    for c in constraints:c['remaining']=max(0,c['limit']-c['current'])
    effective=min([host_available]+[c['remaining'] for c in constraints])
    return dict(host_available_bytes=host_available,effective_available_bytes=effective,process_private_bytes=private,constraints=constraints,unavailable=unavailable,scope='Visible limits plus optional externally verified process ceiling; unknown ancestors remain explicit. Commit/address-space and RAM are distinct,not a portable hardRSS proof')


def memory_allowance(requested,available):
    if type(available) is not int or available<0:raise ValueError('Nonnegative available memory observation')
    if not available:raise MemoryError('No available RAM/commit/address-space headroom')
    if requested=='auto':return max(1,available//2)
    if type(requested) is not int or requested<1:raise ValueError('Positive byte allowance or auto')
    return min(requested,available)


def fit_plan(rows,features,classes,params,mutations,trials,refit,available,time_budget,memory_budget):
    if any(type(v) is not int or v<1 for v in (rows,features,classes)):raise ValueError('Positive shape/class dimensions')
    m=params.n_estimators
    # Worst of the two families; fresh diagnosis after every accepted parent.
    # OOF consists of two half-TRAIN fits per member, never a new size sample.
    native=1+m*(1+3*mutations+trials+int(refit))
    scalar_classes=classes if classes>2 else 1
    scalar=m*(2+3*mutations+trials+int(refit))*scalar_classes
    row_passes=m*((2+2*mutations+trials)*.75*rows+int(refit)*rows)*scalar_classes
    depth=min(params.max_depth,params.max_leaves-1)
    tree_work=row_passes*features*depth
    # Conservative sorting envelope even when bounded prepared bins hit.
    sorting_work=rows*features*math.log2(max(2,rows))*native
    try:automatic=max(150.,30.+tree_work/750000.+sorting_work/40000000.)
    except OverflowError:raise ValueError('Workload outside finite Python time representation') from None
    seconds=automatic if time_budget=='auto' else float(time_budget)
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite time allowance')
    memory=memory_allowance(memory_budget,available);cache=min(32*1024**2,memory//8)
    minimum=rows*features*10+rows*(256+80*classes)+features*512+cache+scalar*(2*params.max_leaves-1)*1024
    return dict(schema='SCALABLE_RESOURCE_PLAN_1',rows=rows,features=features,classes=classes,
                memory_mode='auto' if memory_budget=='auto' else 'explicit',available_bytes=available,
                memory_bytes=memory,estimated_workspace_bytes=minimum,prepared_cache_bytes=cache,
                time_mode='auto' if time_budget=='auto' else 'explicit',seconds=seconds,
                estimated_native_fit_ceiling=native,estimated_scalar_tree_ceiling=scalar,
                scalar_row_passes=row_passes,tree_work_units=tree_work,sort_work_units=sorting_work,
                automatic_seconds=automatic,seconds_floor=150.,tree_units_per_second=750000.,sort_units_per_second=40000000.,
                guarantee='Heuristic forecast plus observed resource checks; not a hard peak or completion guarantee')


def inference_seconds(rows,features,scalar_trees,depth):
    try:seconds=max(150.,30.+rows*(features+scalar_trees*depth)/750000.)
    except OverflowError:raise ValueError('Inference workload outside finite time representation') from None
    if not math.isfinite(seconds):raise ValueError('Finite inference workload required')
    return seconds


def io_seconds(nbytes):return max(150.,nbytes/(1024**2))


def numeric_capacity_plan(plan,params,mutations,trials,refit,explicit_estimators):
    """Derive a finite final numeric domain without using any model score.

    Initial screening and structural evolution keep their original capacity.
    At most45percent of the initial configured time allowance funds final
    numeric candidates/refit. Rates are the existing conservative work model,
    not measured host speed; seeds/shape/explicit allowances are reproducible.
    Memory includes retained candidate models.4096 is the existing HyperParams
    numerical contract,not an admission limit on input observations/features.
    """
    result=dict(plan);base=params.n_estimators;k=plan['classes'] if plan['classes']>2 else 1;n=plan['rows'];f=plan['features'];depth=min(params.max_depth,params.max_leaves-1)
    per_seconds=n*f*depth*k*(.75*trials+int(refit))/plan['tree_units_per_second']+n*f*math.log2(max(2,n))*(trials+int(refit))/plan['sort_units_per_second']
    per_bytes=(trials+int(refit))*((2*params.max_leaves-1)*1024*k+f*params.max_bins*64)
    maximum=base;reason='Explicit estimator capacity' if explicit_estimators else 'Numeric tuning disabled'
    if trials and not explicit_estimators:
        time_max=max(base,math.floor(.45*plan['seconds']/max(1e-12,per_seconds)))
        memory_max=base+max(0,plan['memory_bytes']-plan['estimated_workspace_bytes'])//max(1,per_bytes)
        maximum=max(base,min(4096,time_max,memory_max));reason='Input shape and configured time/memory allowances only'
    result['numeric_capacity']=dict(minimum_estimators=base,maximum_estimators=maximum,reason=reason,score_or_target_values_used=False,time_fraction=.45,seconds_per_estimator_all_numeric_and_refit=per_seconds,bytes_per_additional_estimator_all_candidates=per_bytes)
    result['estimated_native_fit_ceiling']=1+base*(1+3*mutations)+(trials+int(refit))*maximum
    result['estimated_scalar_tree_ceiling']=(base*(2+3*mutations)+(trials+int(refit))*maximum)*k
    result['estimated_workspace_bytes']+=(maximum-base)*per_bytes
    result['scalar_row_passes']=(base*(2+2*mutations)*.75*n+maximum*(.75*trials+int(refit))*n)*k
    result['tree_work_units']=result['scalar_row_passes']*f*depth
    result['sort_work_units']=n*f*math.log2(max(2,n))*result['estimated_native_fit_ceiling']
    result['expanded_work_forecast_seconds']=max(150.,30.+result['tree_work_units']/plan['tree_units_per_second']+result['sort_work_units']/plan['sort_units_per_second'])
    # The selected numerical domain is fixed above. Do not recalculate it
    # from an extended deadline:that would create a capacity/time feedback loop.
    # Auto time must cover the work envelope it selected;explicit deadlines
    # remain exact caller policy and can still stop a best-effort fit early.
    initial_seconds=result['seconds']
    extension=0.
    if plan.get('time_mode')=='auto' and maximum>base:
        extension=max(0.,result['expanded_work_forecast_seconds']-initial_seconds)
        if extension:
            result['seconds']=result['expanded_work_forecast_seconds']
            result['automatic_seconds']=result['seconds']
    result['numeric_capacity']['initial_time_allowance_seconds']=initial_seconds
    result['numeric_capacity']['automatic_deadline_extension_seconds']=extension
    result['numeric_capacity']['domain_recomputed_after_extension']=False
    result['numeric_capacity']['forecast_within_configured_time']=result['expanded_work_forecast_seconds']<=result['seconds'] or math.isclose(result['expanded_work_forecast_seconds'],result['seconds'],rel_tol=1e-12,abs_tol=1e-12)
    return result
