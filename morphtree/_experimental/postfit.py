"""Existing safe-loader/V3/standalone sequence for a returned installed model.

No training or selector here; actual pre-save observations are persisted first.
"""
import json
import time
from pathlib import Path
from .admission import digest,sha
from .runtime_io import artifact_path,reference
from .base import Base
from .target import _NamedRows
from . import roundtrip_v3 as v3


def finish(session,model,root,raw):
    session.check();query={'rows':raw['predict_X'],'names':raw['names']}
    X=_NamedRows(query['rows'],query['names']);obj=model._variant('selected')
    def behavior(m):
        value={'prediction':m.predict(X)}
        if m.task_!='regression':value['probabilities']=m.predict_proba(X)
        return value
    before=behavior(obj);t0=time.perf_counter_ns();source_sha=digest(session.source)
    pre=root/'PRE-SAVE-WITNESS.json'
    pre.write_text(json.dumps({'before':before,'query':query,'source_sha256':source_sha,'captured_before_save':t0},sort_keys=True)+'\n',encoding='utf8')
    p=root/'NATIVE-BEFORE.json';obj.save(p);t1=time.perf_counter_ns()
    loaded=Base.load(p);t2=time.perf_counter_ns();after=behavior(loaded);t3=time.perf_counter_ns()
    q=root/'NATIVE-AFTER.json';loaded.save(q)
    a,b=p.read_bytes(),q.read_bytes();sa,sb=json.loads(a),json.loads(b)
    runtime=session.engine.identity;recipe=digest(obj.best_recipe_);loader=sha(session.engine.package/'model.py')
    witness={'schema':'morph032-actual-behavior-v1','runtime':runtime,'source_sha256':source_sha,'recipe_sha256':recipe,
        'query':query,'loader_sha256':loader,'before_state_sha256':digest(sa),'after_state_sha256':digest(sb),
        'before':before,'after':after,'events':dict(zip(('captured_before_save','saved','safe_loaded','captured_after_load'),(t0,t1,t2,t3)))}
    trust={'runtime':runtime,'actual_runtime':runtime,'before_sha256':sha(p),'after_sha256':sha(q),'witness_sha256':digest(witness),
        'query_sha256':digest(query),'source_sha256':source_sha,'recipe_sha256':recipe,'loader_sha256':loader,
        'admission':('QUALIFIED_RETAINED_ZERO_FIT' if session.mode=='PUBLIC_POSTFIT_IO' else
            'QUALIFIED_COMP003_LIVE_COMPONENTS' if session.mode=='LIVE' else 'QUALIFIED_COMP003_SOFTWARE')}
    comparison=v3.compare(a,b,trust=trust,witness=witness)
    (root/'V3.json').write_text(json.dumps({'result':comparison,'witness':witness,'trust':trust},sort_keys=True,indent=2)+'\n',encoding='utf8')
    if comparison['status']!='PASS':raise ValueError('V3 failed: '+str(comparison))
    path=root/'MODEL.json';model.save(path)
    from .integrated import MorphTree
    restored=MorphTree.load(path,sha256=sha(path),workdir=session.root)
    if behavior(restored)!=before or restored.recipe()!=model.recipe():raise ValueError('Actual final wrapper/recipe changed')
    program=root/'PREDICTOR.py';restored.export_python(program)
    from .worker import standalone
    value={'program':{'path':'PREDICTOR.py','sha256':sha(program),'bytes':program.stat().st_size},'expected':before,'query':query}
    result=standalone(root,value,session.check)
    session.check()
    return {'model':reference(path),'program':reference(program),'expected':before,'query':query,
        'V3':reference(root/'V3.json'),'standalone':result,'task':model.task_,'family':model.best_family_,
        'scope':session.mode,'report':model.report(),'counts':session.counters()}
