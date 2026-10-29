"""Compact operation custody, not an authentication or scientific-usefulness claim.

Dataset buffers stay outside JSON. Digests bind typed source columns, schema,
ordered views and actual labels. Call-local completed-model caches never admit
foreign persisted results; chain verification does not authorize budget replay.
"""
import hashlib
import json
import sys
import uuid

MAX_OPERATION_BYTES=1024*1024
ENVELOPE={'receipt_schema','call_id','sequence','previous','sha256'}
FIELDS={
 'CALL_STARTED':{'engine_sha256','scope','requested_task'},
 'RESOURCE_PLAN':{'plan'},
 'INPUT_PREPARED':{'rows','features','seconds','storage_bytes','input_identity'},
 'NATIVE_FIT_STARTED':{'ordinal','phase','rows','source_rows','family','cache_key','component_key','effective_rows_sha256','effective_labels_sha256','seed','params_sha256','preprocessing_sha256','program_sha256','role','reserved_scalar_trees'},
 'NATIVE_FIT_COMPLETED':{'ordinal','phase','seconds','component_key','actual_scalar_trees'},
 'CACHE_HIT':{'phase','family','rows','cache_key'},
 'SCREEN_COMPLETED':{'losses','parent','seconds'},
 'OOF_FOLD':{'parent_sha256','program_sha256','fit_rows','held_rows','disjoint','encoder_sha256','fit_indices_sha256','held_indices_sha256'},
 'DIAGNOSIS':{'parent_sha256','task','OOF_primary','fold_rows','TRAIN_only','fold_encoder_local','features_examined','region_count','profile_sha256','causal_status','seconds'},
 'MUTATION':{'generation','parent_sha256','child_sha256','child_program','parent_SEARCH','child_SEARCH','decision','seconds'},
 'PROGRAM_FROZEN':{'genome_sha256'},
 'OPTUNA_ASK':{'number','genome_sha256'},
 'OPTUNA_COMPLETE':{'number','SEARCH','parameters'},
 'BIN_PREPARATION_CACHE':{'limit_bytes','retained_bytes','entries','hits','misses','bypassed','evictions','peak_retained_bytes','reused_rows','preparation_seconds','integrity_seconds'},
 'FINAL_MODEL':{'input_rows','selection_TRAIN_rows','SEARCH_rows','final_fit_source_rows','refit_full','SEARCH_primary_before_refit','selection_metric_is_independent','nominal_columns','source_storage_bytes','all_rows_accounted'},
 'FIT_STOPPED':{'error_type'},
}


def canonical(v):
    raw=json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':')).encode('utf8')
    if len(raw)>MAX_OPERATION_BYTES:raise ValueError('Operation metadata exceeds finite allowance; data belong in buffers/artifacts')
    return raw


def sha(v):return hashlib.sha256(canonical(v)).hexdigest()
def framed(h,text):
    raw=text.encode('utf8');h.update(len(raw).to_bytes(8,'little'));h.update(raw)


def schema_digest(names,kinds,vocabularies):
    h=hashlib.sha256(b'SCALABLE_TYPED_SCHEMA_1')
    framed(h,'POSITIONAL' if names is None else 'NAMED')
    if names is not None:
        for name in names:framed(h,name)
    for kind,words in zip(kinds,vocabularies):
        framed(h,kind);h.update(len(words).to_bytes(8,'little'))
        for word in words:framed(h,word)
    return h.hexdigest()


def encoder_digest(encoder):return schema_digest(encoder.names,[c['kind'] for c in encoder.columns],[c.get('vocabulary',()) for c in encoder.columns])


def data_identity(table,labels,role):
    if role not in ('USER_INPUT','TRAIN','TRAIN_FULL'):raise ValueError('Explicit learning-source role')
    if len(labels)!=len(table):raise ValueError('Aligned source/labels for identity')
    h=hashlib.sha256(b'SCALABLE_ARRAY_SOURCE_1')
    schema=schema_digest(table.names,table.kinds,table.vocab);framed(h,schema);framed(h,sys.byteorder)
    for c in table.columns:
        framed(h,c.typecode);h.update(len(c).to_bytes(8,'little'));h.update(memoryview(c))
    source=h.hexdigest()
    selected=hashlib.sha256(memoryview(table.indices)).hexdigest() if table.indices is not None else sha({'range_start':0,'range_stop':len(table)})
    return dict(schema='SCALABLE_DATA_ID_1',source_sha256=source,schema_sha256=schema,effective_indices_sha256=selected,labels_sha256=hashlib.sha256(memoryview(labels)).hexdigest(),rows=len(table),features=table.width,role=role,byteorder=sys.byteorder,label_typecode=labels.typecode)


class Ledger:
    def __init__(self):self.call_id=uuid.uuid4().hex;self.previous='0'*64;self.sequence=0
    def emit(self,event,fields):
        if event not in FIELDS or set(fields)!=FIELDS[event]:raise ValueError('Exact operation metadata schema: '+event)
        v=dict(event=event,**fields,receipt_schema='SCALABLE_OPERATION_1',call_id=self.call_id,sequence=self.sequence,previous=self.previous);v['sha256']=sha(v)
        self.previous=v['sha256'];self.sequence+=1;return v


def verify(events):
    if type(events) is not list or not events or len(events)>10000:raise ValueError('Bounded nonempty receipt chain')
    previous='0'*64;call_id=None
    for i,event in enumerate(events):
        if type(event) is not dict or event.get('event') not in FIELDS or set(event)!=FIELDS[event['event']]|ENVELOPE|{'event'}:raise ValueError('Invalid receipt fields')
        if event['receipt_schema']!='SCALABLE_OPERATION_1' or type(event['sequence']) is not int or event['sequence']!=i or event['previous']!=previous:raise ValueError('Broken receipt order')
        if type(event['call_id']) is not str or len(event['call_id'])!=32 or any(c not in '0123456789abcdef' for c in event['call_id']):raise ValueError('Malformed call identity')
        if call_id is None:call_id=event['call_id']
        if event['call_id']!=call_id:raise ValueError('Cross-call receipt replay')
        body={k:v for k,v in event.items() if k!='sha256'}
        if event['sha256']!=sha(body):raise ValueError('Changed operation receipt')
        previous=event['sha256']
    if events[0]['event']!='CALL_STARTED':raise ValueError('Missing call identity origin')
    return dict(status='VERIFIED_HASH_CHAIN',events=len(events),call_id=call_id,final_sha256=previous,completion='FINAL_MODEL' if events[-1]['event']=='FINAL_MODEL' else 'PARTIAL_OR_NONTRAINING_CHAIN',authority='Integrity/custody only, not authorization or independent scientific validity')
