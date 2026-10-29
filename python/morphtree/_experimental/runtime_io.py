"""Installed IO glue, not a learner. Copied only into private MORPH030 wheels."""
import copy
import json
import os
import tempfile
from contextvars import ContextVar
from pathlib import Path
from .admission import CURRENT,AdmissionError,digest,sha,safe_path
from ..temporary import directory as bounded_directory

WORKDIR=ContextVar('morph030_workdir',default=None)


def configure(workdir):
    p=Path(workdir).resolve()
    if not p.is_dir():raise ValueError('Explicit existing working directory required')
    safe_path(p,p);WORKDIR.set(p)


def artifact_path(value):
    root=WORKDIR.get()
    if root is None:raise ValueError('Call configure(workdir) before explicit IO')
    return safe_path(root,value)


def configure_temp():
    p=artifact_path('.temporary');p.mkdir(exist_ok=True)
    # Limited to this explicit experimental process, not Python installation.
    tempfile.tempdir=str(p)
    return p


def reference(p):
    p=artifact_path(p)
    return {'path':p.relative_to(WORKDIR.get()).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}


def publish(path,text,check):
    path=artifact_path(path);pending=artifact_path(str(path)+'.pending')
    if path.exists() or pending.exists():raise FileExistsError('Never replace completed or incomplete evidence')
    with pending.open('x',encoding='utf8') as f:f.write(text);f.flush();os.fsync(f.fileno())
    check()
    if path.exists():raise FileExistsError('Concurrent output')
    os.rename(pending,path)


def require(operation):
    current=CURRENT.get()
    if current is None or current[1]['operation'] not in operation:raise AdmissionError({'status':'BLOCKED','reason':'EXACT_OPERATION_ADMISSION_REQUIRED','operation':list(operation),'permitted_operations':[]})
    engine,r=current
    if engine.stop.exists():raise AdmissionError({'status':'BLOCKED','reason':'STOP_PERMANENT'})
    return engine,r


def explicit_input(path):
    engine,r=require(('load','replay'))
    p=artifact_path(path)
    matches=[v for v in r['components'] if safe_path(engine.q['input_root'],v['path'])==p]
    if len(matches)!=1:raise ValueError('Exact supplied component reference required')
    b=p.read_bytes();v=matches[0]
    import hashlib
    if len(b)!=v['bytes'] or hashlib.sha256(b).hexdigest()!=v['sha256']:raise ValueError('Changed supplied model bytes')
    # Original bounded native loader sees a private, stable copy of these bytes.
    configure_temp();target=artifact_path(Path(bounded_directory())/'input.json');target.write_bytes(b)
    return target
