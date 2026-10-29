"""Local atomic fitted-state checkpoints. Recovery never resumes training."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .persistence import atomic_text, decode_json_bytes, LoadLimits, read_regular_bytes
from .scalable_receipts import verify
from .temporary import directory as bounded_directory


def _safe_directory(path):
    path = Path(path)
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('Checkpoint directory cannot be a link')
    if not path.is_dir():
        raise ValueError('Checkpoint parent directory must exist')
    return path.resolve()


def _workspace(value, resources):
    estimate = 8192
    pending = [iter((value,))]
    count = 0
    while pending:
        try:
            node = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        count += 1
        estimate += 512
        if isinstance(node, str):
            estimate += len(node) * 32
        elif isinstance(node, dict):
            pending.append(iter(node.items()))
        elif isinstance(node, (list, tuple)):
            pending.append(iter(node))
        if count % 128 == 0:
            resources.reserve(estimate, 'Durable checkpoint workspace')
    resources.reserve(estimate, 'Durable checkpoint workspace')
    return estimate


def commit(estimator, model, classes, resources, *, final):
    from .scalable import digest
    if estimator.checkpoint_dir is None:
        return None
    report = estimator.last_attempt_
    events = report['events']
    verify(events)
    if final:
        if report['status'] != 'COMPLETED' or events[-1]['event'] != 'FINAL_MODEL':
            raise ValueError('Final checkpoint requires completed finalization')
    elif digest(model) != report['last_verified_anchor']['model_sha256']:
        raise ValueError('Checkpoint must match verified TRAIN anchor')
    wanted = {event['parent_sha256'] for event in events if event['event'] == 'DIAGNOSIS'}
    evidence = getattr(estimator, 'diagnostic_profiles_', {})
    resources.reserve(8192, 'Refit configuration header before copying')
    constructor = estimator._constructor_state()
    template = dict(model=model, classes=classes, constructor=constructor, constructor_sha256=digest(constructor), report={k: v for k, v in report.items() if k != 'checkpoint' and (final or k != 'selection_checkpoint')}, evidence=dict(diagnostic_profiles={key: evidence[key] for key in wanted}))
    estimate = _workspace(template, resources)
    packet = copy.deepcopy(template)
    packet['schema'] = 'SCALABLE_PRIVATE_MODEL_3'
    if not final:
        packet['report']['status'] = 'PRESERVED_VERIFIED_TRAIN_ANCHOR_NOT_FINAL_REFIT'
    packet['receipt_sha256'] = digest(packet['report'])
    text = json.dumps(packet, sort_keys=True, allow_nan=False) + '\n'
    resources.reserve(estimate + len(text) * 8, 'Encoded durable checkpoint workspace')
    directory = getattr(estimator, '_active_checkpoint_dir_', None)
    if directory is None:
        if estimator.checkpoint_dir == 'auto':
            parent = _safe_directory(tempfile.gettempdir()) / 'morphtree-checkpoints'
            parent.mkdir(exist_ok=True)
        else:
            parent = Path(estimator.checkpoint_dir)
        parent = _safe_directory(parent)
        directory = Path(bounded_directory(prefix='fit-', dir=parent))
        estimator._active_checkpoint_dir_ = directory
        estimator._checkpoint_sequence_ = 0
    directory = _safe_directory(directory)
    report['checkpoint_directory'] = str(directory)
    if shutil.disk_usage(directory).free < len(text) * 4 + 8192:
        raise OSError('Insufficient free disk for atomic checkpoint; previous model preserved')
    name = f"model-{estimator._checkpoint_sequence_:04d}.json"
    target = directory / name
    if target.exists():
        raise FileExistsError('An existing fitted checkpoint cannot be overwritten')
    atomic_text(target, text)
    record = dict(schema='SCALABLE_CHECKPOINT_POINTER_1', path=name, sha256=hashlib.sha256(text.encode('ascii')).hexdigest(), call_id=events[0]['call_id'], source_role='TRAIN_FULL' if final and estimator.refit_full else 'TRAIN', final=final)
    if not final:
        atomic_text(directory / 'SELECTED.json', json.dumps(record, sort_keys=True) + '\n')
    atomic_text(directory / 'CURRENT.json', json.dumps(record, sort_keys=True) + '\n')
    estimator._checkpoint_sequence_ += 1
    receipt = dict(directory=str(directory), **record)
    report['checkpoint'] = receipt
    if not final:
        report['selection_checkpoint'] = receipt
    if final:
        estimator.checkpoint_path_ = str(target)
    return receipt


def read_pointer(directory, resources, *, stage='current'):
    if type(stage) is not str or stage not in ('current', 'selected'):
        raise ValueError('Checkpoint stage must be current or selected')
    directory = _safe_directory(directory)
    pointer = directory / ('CURRENT.json' if stage == 'current' else 'SELECTED.json')
    if pointer.is_symlink():
        raise ValueError('Checkpoint pointer cannot be a link')
    raw = read_regular_bytes(pointer,8192)
    record = decode_json_bytes(raw, LoadLimits(max_bytes=8192), reserve=resources.reserve)
    required = {'schema', 'path', 'sha256', 'call_id', 'source_role', 'final'}
    if set(record) != required or record['schema'] != 'SCALABLE_CHECKPOINT_POINTER_1':
        raise ValueError('Exact checkpoint pointer schema required')
    name = record['path']
    if type(name) is not str or len(name) != 15 or not name.startswith('model-') or not name.endswith('.json') or not name[6:10].isdigit():
        raise ValueError('Checkpoint pointer must name a local numbered model')
    if type(record['sha256']) is not str or len(record['sha256']) != 64 or any(c not in '0123456789abcdef' for c in record['sha256']) or type(record['call_id']) is not str or not record['call_id'] or type(record['final']) is not bool or record['source_role'] not in ('TRAIN', 'TRAIN_FULL') or not record['final'] and record['source_role'] != 'TRAIN':
        raise ValueError('Invalid checkpoint identity or source role')
    if stage == 'selected' and record['final']:
        raise ValueError('Selected checkpoint must be a verified TRAIN anchor')
    target = directory / name
    if target.is_symlink() or target.resolve().parent != directory:
        raise ValueError('Checkpoint model must be a local regular file')
    return target, record
