"""Independent metadata/slot replay of recorded ordinary eviction transitions."""
import base64,gzip,hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from native_controller import kernel
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def decode(record):
    dtype=np.dtype(record['dtype']);shape=tuple(record['shape'])
    raw=base64.b64decode(record['data'],validate=True)
    assert len(raw)==np.prod(shape,dtype=np.int64)*dtype.itemsize
    return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()


def exact(expected,record):
    actual=decode(record)
    assert expected.shape==actual.shape and expected.dtype==actual.dtype
    assert expected.tobytes()==actual.tobytes()


def main():
    out=BASE/'verified_corpus.json';assert not out.exists()
    plan=read(BASE/'plan.json');source=read(BASE/'source.json');report=read(BASE/'capture.json')
    assert report['passed'] and report['frozen50predictions_state_counters_payload_calls_replayed']
    assert report['hooks_restored'] and report['owned_host_after_cleanup_bytes']==0
    assert source['plan_sha256']==report['plan_sha256']==sha(BASE/'plan.json') and source['archive_sha256']==sha(BASE/'source.tar.gz')
    with tarfile.open(BASE/'source.tar.gz') as t:
        archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
    assert archived==source['sources'] and all(sha(ROOT/f)==h for f,h in archived.items())
    assert read(BASE/'capture_job.json')['status']=='finished' and read(BASE/'capture_job.json')['exit_code']==0
    assert sha(ROOT/plan['reference'])==plan['reference_sha256'];reference=read(ROOT/plan['reference'])
    assert report['state_sha256']==reference['state_sha256'] and report['counters']==reference['counters']
    assert report['profiled_timing_not_latency_gate']['trials'][0]['predicted_token_ids']==reference['predicted_token_ids']
    for f,v in report['payload'].items():assert v==reference['prototype_counts'][f]
    for f in ['native_h2d_calls','native_d2h_calls']:assert report[f]==reference['prototype_counts'][f]
    corpus=BASE/'transitions.jsonl.gz';assert sha(corpus)==report['corpus_sha256']
    counts=dict(records=0,paid_rows=0,restored_backups=0,cold_overflow_records=0,queued_compactions=0)
    identities=set();cold_max=0
    with gzip.open(corpus,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);before=row['before'];after=row['after'];cfg=row['config']
            identity=(row['step'],row['layer']);assert identity not in identities;identities.add(identity)
            assert cfg['inclusive'] and cfg['batch_backup_updates']
            new=row['new_positions'];new=[new] if isinstance(new,int) else new
            assert len(new)==row['new_rows']
            hot=before['hot_positions']+new;victims=row['prepared_victims']
            assert row['prepared_positions']==hot and victims is not None and len(victims)==len(set(victims))
            assert all(0<=i<len(hot) for i in victims)
            shadow=decode(before['stt_shadow']);resident=decode(before['stt_was_vram_resident']);streak=decode(before['stt_peek_streak']);timestamps=decode(before['stt_last_access'])
            assert shadow.dtype==resident.dtype==np.bool_ and streak.dtype==timestamps.dtype==np.float32
            cold=before['cold_positions'];assert len(cold)==len(shadow)==len(resident)==len(streak)==len(timestamps)
            lookup={p:i for i,p in enumerate(cold) if shadow[i]}
            paid=[i for i in victims if hot[i] not in lookup];restored=[lookup[hot[i]] for i in victims if hot[i] in lookup]
            native=kernel().backups([dict(hot=hot,cold=cold,shadow=shadow,victims=victims,inclusive=True)])[0]
            assert native==dict(paid=paid,restored=restored)
            shadow[restored]=False;streak[restored]=0;timestamps[restored]=row['step']
            paid_count=len(paid)
            cold=cold+[hot[i] for i in paid]
            assert len(cold)<=cfg['sttram_capacity'],'Saturated transitions require a distinct replay rule; never silently skip them'
            cold_max=max(cold_max,len(cold))
            shadow=np.concatenate((shadow,np.zeros(paid_count,dtype=np.bool_)))
            resident=np.concatenate((resident,np.ones(paid_count,dtype=np.bool_)))
            streak=np.concatenate((streak,np.zeros(paid_count,dtype=np.float32)))
            timestamps=np.concatenate((timestamps,np.full(paid_count,row['step'],dtype=np.float32)))
            assert after['cold_positions']==cold
            for name,value in [('stt_shadow',shadow),('stt_was_vram_resident',resident),('stt_peek_streak',streak),('stt_last_access',timestamps)]:exact(value,after[name])
            keep=np.flatnonzero(~np.isin(np.arange(len(hot)),victims)).tolist()
            assert after['hot_positions']==[hot[i] for i in keep]
            assert after['logical_slots']==before['logical_slots']+before['free_slots'][:paid_count]
            assert after['free_slots']==before['free_slots'][paid_count:]
            slots=after['logical_slots']+after['free_slots'];assert len(slots)==len(set(slots)) and len(after['logical_slots'])==len(cold)
            assert row['outcome']==[len(victims),0,len(restored),0]
            exact(np.zeros(len(keep),dtype=np.bool_),after['sram_grace'])
            pending=after['pending_keep']
            if pending is not None:assert pending==keep;counts['queued_compactions']+=1
            for name in ['sram_cum','sram_age']:
                value=np.concatenate((decode(before[name]),np.zeros(row['new_rows'],dtype=np.float32)))
                if pending is None:value=value[keep]
                exact(value,after[name])
            counts['records']+=1;counts['paid_rows']+=paid_count;counts['restored_backups']+=len(restored)
    assert identities=={(step,layer) for step in range(1,51) for layer in range(32)}
    assert counts['records']==report['transition_records']==1600
    result=dict(verified=True,counts=counts,recorded_cold_max=cold_max,configured_cold_capacity=1090,frozen_model_replay_exact=True,independent_metadata_and_slot_replay_exact=True,native_backup_classification_exact=True,ready_for_cpu_transition_layout_stage=True,full_goal_complete=False,limits=report['corpus_limits']+' No cold overflow observed; saturated/nonfinite/tie/invalid-input and promotion-selection fixtures remain required. Replay preserves raw F32 bytes, CPU metadata, slot/free ordering and queued compaction indices. It is not a new controller, speed or quality claim.',evidence_sha256={f:sha(BASE/f) for f in ['plan.json','source.json','source.tar.gz','capture.py','capture.json','capture_job.json','capture.log','transitions.jsonl.gz','verify_corpus.py']})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
