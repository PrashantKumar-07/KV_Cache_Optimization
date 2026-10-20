"""Development falsification of pooled-head maximum placement at fixed budgets."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from check_batched_boundaries import measure
from head_max_retention import use_head_max
from provenance import atomic_json,runtime_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    source=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))
    report=dict(passed=False,runtime=runtime_identity(),rows=[],manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    try:
        manifest=json.loads(args.manifest.read_text());cases=[c for c in manifest['cases'] if c.get('oracle')]
        prior_path=Path('results/baseline_fidelity/tiered_query_needles_v2_20261001.json')
        prior=json.loads(prior_path.read_text());assert prior['passed'] and len(cases)==6
        report['prior_sha256']=hashlib.sha256(prior_path.read_bytes()).hexdigest()
        control={r['case']:r for r in prior['rows'] if r['arm']=='actual_query'}
        assert len(control)==6
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained(manifest['model'],torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'])
        report['gpu']=torch.cuda.get_device_name()
        for index,case in enumerate(cases):
            assert hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()==case['token_sha256']
            pair={}
            for arm in (('sum_heads','max_head') if index%2==0 else ('max_head','sum_heads')):
                with use_head_max() if arm=='max_head' else nullcontext():
                    result=measure(model,tokenizer,case,'gpu',prompt_queries=True)
                if arm=='sum_heads':
                    for field in ('generated_token_ids','prediction','state_sha256','trajectory_sha256',
                                  'trajectory_frames','initial_answer_retention','counters','prefill_query_shapes'):
                        assert json.loads(json.dumps(result[field]))==control[case['name']][field],field
                complete=sum(x['retained_tokens']==x['span_tokens'] for x in result['initial_answer_retention'].values())
                pair[arm]=result
                report['rows'].append(dict(case=case['name'],arm=arm,token_sha256=case['token_sha256'],complete_span_layers=complete,**result))
                atomic_json(args.output,report)
                print(case['name'],arm,result['answer_correct'],'complete layers',complete,flush=True)
            assert pair['sum_heads']['initial_cache_rows']==pair['max_head']['initial_cache_rows']
        correct={arm:sum(r['answer_correct'] for r in report['rows'] if r['arm']==arm) for arm in ('sum_heads','max_head')}
        retained={arm:sum(r['complete_span_layers'] for r in report['rows'] if r['arm']==arm) for arm in correct}
        report.update(passed=True,correct_cases=correct,complete_span_layers=retained,
            advance_to_broader_development=correct['max_head']>=4 and retained['max_head']>retained['sum_heads'],
            limitations='Six reused synthetic development probes, one model/seed; sameinitialgeometry, finaltrajectory differs. Pooling/headreduction changes initialcum and subsequenteviction aswellas promptretention. No independentquality/physicalresources/SnapKV/latency/defaultpromotion claim.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:atomic_json(args.output,report)


if __name__=='__main__':main()
