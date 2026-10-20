"""Frozen six-task free-generation preservation; no new quality superiority claim."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bounded_pinned_pool import BoundedPoolPolicy
from direct_shared_pool import DirectScorePolicy
from check_batched_generation import Trajectory, measure
from provenance import atomic_json, read_jsonl, runtime_identity


class Control(Trajectory,DirectScorePolicy):
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)


class Candidate(Trajectory,BoundedPoolPolicy):
    def __init__(self,**options):
        self.trajectory,self.frames=hashlib.sha256(),0
        super().__init__(**options)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    m=json.loads(args.manifest.read_text())
    hot,cold=m['hot'],m['cold']
    expected_reservation=dict(direct=(cold+hot+1)*131072,
        bounded=(cold+32+m['max_append'])*131072)
    assert m['max_append']==32
    report=dict(passed=False,runtime=runtime_identity(),rows=[],manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))})
    try:
        requests=[]
        for task,meta in m['tasks'].items():
            source=Path(meta['source_file']);assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
            data=read_jsonl(source)
            for index,sample in zip(meta['selected_zero_based_indices'],meta['selected_sample_ids']):
                item=data[index];assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
                requests.append((task,item,sample))
        assert len(requests)==12 and m['seeds'] in ([0],[0,1])
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(m['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(m['model'])
        report['gpu']=torch.cuda.get_device_name()
        for seed in m['seeds']:
            for index,(task,item,sample) in enumerate(requests):
                pair={}
                order=('direct','bounded') if (index+seed)%2==0 else ('bounded','direct')
                for arm in order:
                    result=measure(model,tokenizer,item,task,seed,'cpu_scores',31500,prompt_queries=True,
                        policy_cls=Control if arm=='direct' else Candidate,
                        policy_options=dict(sram_budget=hot,stt_budget=cold,stt_expose_quota=32))
                    assert result['reserved_host_tensor_bytes']==expected_reservation[arm]
                    pair[arm]=result
                    report['rows'].append(dict(task=task,sample_id=sample,seed=seed,arm=arm,**result))
                    atomic_json(args.output,report)
                    print(seed,index+1,task,arm,len(result['generated_token_ids']),flush=True)
                for field in ('prediction','generated_token_ids','score','generation','max_gen','termination','prompt_sha256',
                    'state_sha256','trajectory_sha256','trajectory_frames','counters','prefill_query_shapes',
                    'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','occupied_host_tensor_bytes'):
                    assert pair['direct'][field]==pair['bounded'][field],field
        assert len(report['rows'])==24*len(m['seeds'])
        report.update(passed=True,exact_generation_state_trajectory_and_payload=True,
            limitations='Reused development documents, six tasks/two docs/greedy generation; seeds recorded in manifest. Integrity preservation only; seed does not add sampling diversity. No independent quality superiority, latency, serving, physicalSTT or broadcompletion claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)


if __name__=='__main__':main()
