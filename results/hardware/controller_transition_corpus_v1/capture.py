"""Capture ordinary CPU/controller transitions; diagnostic, never latency evidence."""
import base64,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from bulk_promotion_policy import BulkPromotionPolicy
from profile_cpu_batched import ProfileCalls
BASE=Path(__file__).resolve().parent

def array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().contiguous().numpy()
    a=np.ascontiguousarray(value)
    return dict(dtype=a.dtype.str,shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode('ascii'))


def snapshot(cache):
    result={name:array(getattr(cache,name)) for name in ['sram_cum','sram_age','sram_grace','stt_shadow','stt_was_vram_resident','stt_peek_streak','stt_last_access']}
    result.update(hot_positions=list(cache.sram_pos),cold_positions=list(cache.stt_pos),logical_slots=list(cache._logical),free_slots=list(cache._free),pending_keep=cache.pending_keep.tolist() if cache.pending_keep is not None else None)
    return result


class CapturePolicy(ProfileCalls,BulkPromotionPolicy):
    records=0
    sink=None
    originals=None
    def _bifurcate(self,past):
        result=super()._bifurcate(past)
        self.originals=[]
        for layer,c in self.layer_caches.items():
            original=c.evict_and_demote
            self.originals.append((c,original))
            def wrapped(*args,_layer=layer,_cache=c,_original=original,**options):
                before=snapshot(_cache)
                prepared=_cache.prepared_victims
                record=dict(layer=_layer,step=_cache._t,new_positions=options.get('new_pos'),new_rows=args[0].shape[1],prepared_positions=list(prepared[0]) if prepared else None,prepared_victims=list(prepared[1]) if prepared else None,before=before,config={name:getattr(_cache.cfg,name) for name in ['sram_capacity','sttram_capacity','inclusive','batch_backup_updates','sink_size','window_size','stt_live_floor']})
                outcome=_original(*args,**options)
                record.update(outcome=list(outcome),after=snapshot(_cache))
                type(self).sink.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                type(self).records+=1
                return outcome
            c.evict_and_demote=wrapped
        return result
    def reset(self):
        for cache,method in self.originals or []:cache.evict_and_demote=method
        self.originals=[]
        super().reset()


def main():
    output=BASE/'capture.json';corpus=BASE/'transitions.jsonl.gz';partial=BASE/'transitions.partial.jsonl.gz'
    assert not any(p.exists() for p in [output,corpus,partial])
    source=(ROOT/'results/hardware/completion_critical_path_v1/profile_ordinary_v2.py').read_text()
    for old,new in [("BASE/'profile_v2.json'","BASE/'capture.json'"),("BASE/'profile_plan_v2.json'","BASE/'plan.json'"),("BASE/'policy_v2.prof'","BASE/'capture.prof'"),("BASE/'policy_v2.prof.txt'","BASE/'capture.prof.txt'")]:
        assert old in source;source=source.replace(old,new)
    scope=dict(__name__='capture_controller_transitions',__file__=str(Path(__file__)))
    exec(compile(source,'<ordinary-controller-transition-capture>','exec'),scope)
    scope['ProfileBulk']=CapturePolicy
    CapturePolicy.records=0
    with gzip.open(partial,'wt',encoding='utf-8') as sink:
        CapturePolicy.sink=sink
        try:scope['main']()
        finally:CapturePolicy.sink=None
    assert CapturePolicy.records==1600
    partial.rename(corpus)
    report=json.loads(output.read_text());report.update(transition_records=CapturePolicy.records,corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),capture_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),corpus_limits='Post-promotion/pre-eviction input and post-eviction output, before visibility compaction. Prepared victim decisions are fixed inputs; this corpus does not independently validate selection/promotion. F32 stored as exact raw bytes including nonfinite values. No KV contents or hardware-performance claim.')
    output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
