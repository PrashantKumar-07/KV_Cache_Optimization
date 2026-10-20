"""Paired development quality for native-pool current-query selection."""
import gc,hashlib,json
from pathlib import Path
import sys
from functools import partial
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import check_batched_generation as generation
from check_batched_generation import Trajectory,measure
from bulk_promotion_policy import BulkPromotionPolicy
from pool_query_selector import PoolQuerySelector
from audit_unused_documents import document_key
from provenance import atomic_json,read_jsonl,runtime_identity


class Observed(Trajectory,BulkPromotionPolicy):
    model=None
    query_enabled=False
    last_observed=None
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)
        self.selector=PoolQuerySelector(type(self).model,self) if type(self).query_enabled else None
    def reset(self):
        type(self).last_observed=dict(query_enabled=self.selector is not None,
            selected_layers=self.selector.selected_layers if self.selector else 0,
            scanned_rows=self.selector.scanned_rows if self.selector else 0)
        if self.selector:self.selector.close()
        super().reset()


def main():
    base=Path(__file__).resolve().parent;output=base/'development.json';assert not output.exists()
    p=json.loads((base/'development_plan.json').read_text());m=json.loads(Path(p['development_manifest']).read_text())
    assert hashlib.sha256(Path(p['development_manifest']).read_bytes()).hexdigest()==p['development_manifest_sha256']
    requests=[]
    for task,entry in m['tasks'].items():
        source=Path(entry['source_file']);assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['source_sha256']
        items=read_jsonl(source)
        for index,sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
            item=items[index];assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
            requests.append((task,sample,item,document_key(item)))
    assert len(requests)==12 and len({r[3] for r in requests})==12
    holdout=json.loads(Path('results/baseline_fidelity/snap_tier_envelope_fresh_quality_20261002/manifest.json').read_text())
    held_contexts={key for meta in holdout['tasks'].values() for key in meta['selected_normalized_context_sha256']}
    assert not held_contexts.intersection(r[3] for r in requests)
    report=dict(passed=False,runtime=runtime_identity(),rows=[],development_plan_sha256=hashlib.sha256((base/'development_plan.json').read_bytes()).hexdigest(),
                holdout_context_overlap=0,expected_rows=24)
    original=generation.bench.generate_with_policy
    generation.bench.generate_with_policy=partial(original,prefill_last_logits=True)
    try:
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(m['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(m['model'])
        for number,(task,sample,item,context) in enumerate(requests):
            for arm in (('control','candidate') if number%2==0 else ('candidate','control')):
                cold=p['cold'];Observed.model=model;Observed.query_enabled=arm=='candidate'
                before=[len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
                result=measure(model,tokenizer,item,task,0,'cpu_scores',p['max_ctx'],prompt_queries=True,
                    policy_cls=Observed,policy_options=dict(sram_budget=p['hot'],stt_budget=cold,stt_expose_quota=p['quota']))
                assert result['reserved_host_tensor_bytes']==(cold+64)*131072
                assert before==[len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers], 'Selector hooks leaked'
                if arm=='candidate':assert Observed.last_observed['selected_layers']>0
                result['selector_hooks_removed']=True
                report['rows'].append(dict(task=task,sample_id=sample,context_sha256=context,arm=arm,
                    observed=Observed.last_observed,**result))
                atomic_json(output,report)
                print(number+1,'/12',task,arm,result['score'],len(result['generated_token_ids']),flush=True)
        assert len(report['rows'])==24
        report.update(passed=True,limits='Reused12development requests/one seed, same real model and ordinary bulk generator. '
            'Candidate uses full-scan current-query selection; no independent quality superiority, fair exact usage, speed or physicalSTT claim.')
    except BaseException as e:report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        generation.bench.generate_with_policy=original
        gc.collect();atomic_json(output,report)


if __name__=='__main__':main()
