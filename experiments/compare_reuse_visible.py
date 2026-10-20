"""Reuse the existing recorded model trial for the native storage candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import compare_cpu_batched as trial
from native_reuse_visible import NativeReusePolicy,kernel
from provenance import atomic_json,runtime_identity


class RecordedNativePolicy(trial.Recording,NativeReusePolicy):
    last_counts={}

    def __init__(self,**options):
        super().__init__(**options)
        self.history=[];self.state_history=[];self.record_full_state=False

    def reset(self):
        type(self).last_counts=dict(reused_layers=self.reused_layers,
            fallback_layers=self.reuse_fallback_layers,
            reused_compactions=sum(getattr(c,'reused_compactions',0) for c in self.layer_caches.values()))
        super().reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contexts',nargs='+',type=int,default=[4096,8192,16384,31500])
    parser.add_argument('--decode',type=int,default=12)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--blocks',type=int,default=0)
    parser.add_argument('--hot',type=int,default=991)
    parser.add_argument('--cold',type=int,default=1090)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    files=list(Path('experiments').glob('*.py'))+list(Path('src').glob('*.py'))+[
        Path('experiments/native_hot_compact.cpp'),Path('experiments/native_hot_compact.cu'),Path('experiments/native_reuse_hot.cu')]
    report=dict(passed=False,config=vars(args)|{'output':str(args.output)},rows=[],runtime=runtime_identity(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    original=trial.FusedBoundedPolicy
    try:
        report['reuse_binary_sha256']=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest()
        prompt_path=Path('results/hardware/batched_eviction_coordinator_20261001/max_context_prompt.txt')
        teacher_path=Path('results/shadow_ablation_full/fixed_stream_20260929.json')
        prompt=prompt_path.read_text();stream=json.loads(teacher_path.read_text())['config']['teacher_token_ids']
        report.update(prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            teacher_source_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),teacher_token_ids=stream[:args.decode+1])
        torch.manual_seed(0);model_path='../models/mistral-7b-instruct'
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained(model_path);report['gpu']=torch.cuda.get_device_name()
        for context in args.contexts:
            reference=None
            for block in range(args.blocks or 1):
                arms=('fused_bounded','native_reuse','native_reuse','fused_bounded') if args.blocks else ('fused_bounded','native_reuse')
                for arm in arms:
                    trial.FusedBoundedPolicy=original if arm=='fused_bounded' else RecordedNativePolicy
                    row=trial.measure(model,tokenizer,prompt,stream,context,args.decode,'fused_bounded_scores',
                        args.warmup,args.blocks==0,real_query_prefill=True,hot=args.hot,cold=args.cold)
                    if reference is None:reference=row
                    else:
                        trial.check(reference,row)
                        for field in ('reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
                            assert row[field]==reference[field],field
                    if arm=='native_reuse':
                        row['reuse_counts']=dict(RecordedNativePolicy.last_counts)
                        assert row['reuse_counts']['reused_compactions']>0
                    report['rows'].append(dict(arm=arm,context=context,block=block,**row))
                    atomic_json(args.output,report)
                    print(context,block,arm,'exact',flush=True)
        report.update(passed=True,limitations='Same existing recorded trial/profile synchronization and policyhistory instrumentation. Native path consumes decode buffers, keeps BF16hot views/metadata exact, rootcapacity unchanged. Sourceclass replacement scoped/restored finally. No ordinaryCLI or speed/quality/physicalSTT/memory-superiority claim; lifetime/generation/isolation/ABBA gates remain mandatory.')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        trial.FusedBoundedPolicy=original;atomic_json(args.output,report)


if __name__=='__main__':main()
