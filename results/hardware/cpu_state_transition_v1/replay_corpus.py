"""Replay every recorded CPU phase and its GPU metadata commands, no model run."""
import base64,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from cpu_state import CpuStateArena
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def decode(record):
    raw=base64.b64decode(record['data'],validate=True);dtype=np.dtype(record['dtype']);shape=tuple(record['shape'])
    assert len(raw)==np.prod(shape,dtype=np.int64)*dtype.itemsize
    return np.frombuffer(raw,dtype=dtype).reshape(shape).copy()

def packed(rows,dtype=np.int32):
    values=np.full((len(rows),max(map(len,rows),default=0)),-1,dtype=dtype)
    for i,row in enumerate(rows):values[i,:len(row)]=row
    return values,np.asarray([len(row) for row in rows],dtype=np.int32)

def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main():
    output=BASE/'replay.json';assert not output.exists()
    corpus=ROOT/'results/hardware/controller_transition_corpus_v1/transitions.jsonl.gz'
    proof=json.loads((ROOT/'results/hardware/controller_transition_corpus_v1/verified_corpus.json').read_text())
    assert proof['verified'] and sha(corpus)==proof['evidence_sha256']['transitions.jsonl.gz']
    state=CpuStateArena(32,1055,1154);roots={k:a.ctypes.data for k,a in state.owners.items()}
    totals=dict(records=0,paid_rows=0,restored_backups=0,batches=0)
    with gzip.open(corpus,'rt') as stream:
        for step in range(1,51):
            rows=[json.loads(next(stream)) for _ in range(32)]
            assert [(r['step'],r['layer']) for r in rows]==[(step,l) for l in range(32)]
            for l,row in enumerate(rows):
                before=row['before']
                state.load(l,before['hot_positions'],before['cold_positions'],decode(before['sram_grace']),decode(before['stt_shadow']),decode(before['stt_was_vram_resident']),decode(before['stt_peek_streak']),before['logical_slots'],before['free_slots'])
            new_rows=[[r['new_positions']] if isinstance(r['new_positions'],int) else r['new_positions'] for r in rows]
            state.append_hot(*packed(new_rows,dtype=np.int64))
            for l,r in enumerate(rows):assert state.view('hot',l).tolist()==r['prepared_positions']
            commands=state.demote(*packed([r['prepared_victims'] for r in rows]));state.validate_commands(commands)
            for l,row in enumerate(rows):
                before,after=row['before'],row['after'];paid,restored,kept=map(int,commands['sizes'][l])
                for name,field in [('hot','hot_positions'),('cold','cold_positions'),('slots','logical_slots'),('free','free_slots')]:assert state.view(name,l).tolist()==after[field]
                for name,field in [('grace','sram_grace'),('shadow','stt_shadow'),('resident','stt_was_vram_resident'),('streak','stt_peek_streak')]:exact(state.view(name,l),decode(after[field]))
                assert row['outcome']==[len(row['prepared_victims']),0,restored,0]
                assert commands['keep'][l,:kept].tolist()==after['pending_keep']
                assert commands['writes'][l,:paid].tolist()==before['free_slots'][:paid]
                timestamps=decode(before['stt_last_access']);timestamps[commands['restored'][l,:restored]]=step
                timestamps=np.concatenate((timestamps,np.full(paid,step,np.float32)))
                exact(timestamps,decode(after['stt_last_access']))
                for field in ['sram_cum','sram_age']:
                    projected=np.concatenate((decode(before[field]),np.zeros(row['new_rows'],np.float32)))
                    selected=commands['keep'][l,:kept]
                    # The recorded queued phase keeps physical metadata uncompressed.
                    # Both descriptor gathers must produce the same logical F32 bits.
                    exact(projected[selected],decode(after[field])[after['pending_keep']])
                totals['records']+=1;totals['paid_rows']+=paid;totals['restored_backups']+=restored
            assert roots=={k:a.ctypes.data for k,a in state.owners.items()}
            totals['batches']+=1
        assert stream.read()==''
    assert totals==dict(records=1600,paid_rows=2560,restored_backups=23872,batches=50)
    result=dict(passed=True,counts=totals,owned_cpu_bytes=state.owned_bytes,roots_preserved=True,exact_metadata_and_slots=True,gpu_metadata_commands_exact=True,full_controller_integrated=False,ordinary_adoption=False,full_goal_complete=False,limits='Native CPU phase only. GPU metadata commands applied to recorded raw F32 arrays on CPU; no GPU payload/model integration or speed claim. No cold overflow occurs in this corpus; independent saturation/promotion/reclaim fixtures required. Native container/output/Python allocator workspace is unmeasured.',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'experiments/cpu_state.py',ROOT/'experiments/native_cpu_state.cpp',Path(__file__)]},corpus_sha256=sha(corpus),corpus_proof_sha256=sha(ROOT/'results/hardware/controller_transition_corpus_v1/verified_corpus.json'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
