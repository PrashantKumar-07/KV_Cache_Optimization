"""Observe expanded SnapKV decode storage lifetimes with the same weak tracker."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import audit_decode_kv_lifetimes as lifetime
from audit_snap_tier_resources import snapshot
from longbench_eval import _attn_impl
from provenance import atomic_json,runtime_identity
from snapkv_adapter import SnapKVAdapter


class CompactTrace(lifetime.StorageTrace):
    """Retain allocation/expiry transitions and explicit phase samples."""
    def __init__(self):
        super().__init__();self.signature=None;self.observed_samples=0

    def sample(self,operation):
        super().sample(operation);self.observed_samples+=1
        row=self.samples[-1]
        signature=(row['stage'],tuple(sorted(row['known_live_kv_storage_bytes'].items())),
                   row['live_storage_count'],len(self.allocations))
        if operation.startswith('aten.') and signature==self.signature:
            self.samples.pop()
        else:self.signature=signature


class ObservedSnap(SnapKVAdapter):
    active=None

    def update(self,*args,**kwargs):
        if type(self).active is None:return super().update(*args,**kwargs)
        with type(self).active.phase('cache_update'):
            return super().update(*args,**kwargs)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    files=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))+[
        Path('sota/snapkv/snapkv/monkeypatch/snapkv_utils.py')]
    report=dict(passed=False,runtime=runtime_identity(),self_check=lifetime.self_check(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},traces=[],boundaries=[])
    original=lifetime.is_kv;past=None
    try:
        # Query length is1 during this decode-only trace; large expanded rows
        # come from stored/converted KV. Prefill executes outside the tracker.
        lifetime.is_kv=lambda t:original(t) or (t.dtype in (torch.bfloat16,torch.float32)
            and t.ndim==4 and t.shape[0]==1 and t.shape[1]==32 and t.shape[2]>=974 and t.shape[3]==128)
        prompt=Path('results/remediation_v3/latency_prompt.txt')
        teachers=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        stream=json.loads(teachers.read_text())['config']['teacher_token_ids'][:50]
        assert len(stream)==50
        torch.manual_seed(0);model_path='../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(model_path)
        ids=tokenizer(prompt.read_text(),return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384)
        past=ObservedSnap(model,capacity=974,window=32,kernel=5)
        with _attn_impl(model,'sdpa'):
            out=model(ids,past_key_values=past,use_cache=True,position_ids=torch.arange(16384,device='cuda')[None])
        past=out.past_key_values;del out
        report.update(model=model_path,seed=0,context=16384,decode=50,gpu=torch.cuda.get_device_name(),
            prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teachers.read_bytes()).hexdigest(),
            input_token_ids=ids.cpu().tolist()[0],teacher_token_ids=stream)
        report['boundaries'].append(snapshot(past,None,0));predictions=[]
        for step,token in enumerate(stream):
            trace=CompactTrace();trace.stage='model_forward'
            trace.track([(layer.keys,layer.values) for layer in past.layers],'decode_entry')
            trace.sample('decode_entry');ObservedSnap.active=trace
            try:
                with trace:
                    out=model(torch.tensor([[token]],device='cuda'),past_key_values=past,use_cache=True,
                        position_ids=torch.tensor([[16384+step]],device='cuda'))
                    trace.sample('forward_return')
                past=out.past_key_values
                predictions.append(out.logits[:,-1:].argmax(-1).item())
                report['traces'].append(dict(step=step+1,allocations=trace.allocations,
                    samples=trace.samples,observed_samples=trace.observed_samples))
                report['boundaries'].append(snapshot(past,None,step+1));del out
            finally:ObservedSnap.active=None;trace.close()
        report['predicted_token_ids']=predictions
        prior=json.loads(Path('results/baseline_fidelity/snap_tier_owned_envelope_20261002/result.json').read_text())
        reference=next(r for r in prior['rows'] if r['arm']=='snap')
        assert predictions==reference['predicted_token_ids']
        for row,old in zip(report['boundaries'],reference['boundaries']):
            for key in ('step','layers','distinct_owned_kv_storage_bytes','visible_kv_tensor_payload_bytes'):
                assert row[key]==old[key]
        peaks={}
        for trace in report['traces']:
            for sample in trace['samples']:
                stage=sample['stage']
                if stage not in peaks or sample['known_live_kv_total_bytes']>peaks[stage]['known_live_kv_total_bytes']:
                    peaks[stage]=dict(step=trace['step'],**sample)
        report.update(passed=True,phase_peaks=peaks,predictions_and_boundaries_preserved=True,
            limitations='Untimed one-model/seed/fixedstream Snap974expanded/window32/kernel5 decode; same weak-storage tracker, expanded KV rows>=974 also included. Small projected GQA KV tracked, expanded one-row copies/query shapes excluded. Shape-classified known KV/F32 copies only, metadata/model/attention temporaries, retained allocator blocks and freed-stream-pending buffers excluded. This lower bound is not full physical equality or a speed measurement.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        lifetime.is_kv=original;ObservedSnap.active=None
        if past is not None:past.close()
        atomic_json(args.output,report)


if __name__=='__main__':main()
