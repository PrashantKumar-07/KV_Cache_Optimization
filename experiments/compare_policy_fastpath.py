"""ABBA comparison for behavior-preserving TieredKV metadata fast paths."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ("src", "experiments")]
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import runtime_identity


class LegacyMetadataPolicy(TieredKVPolicy):
    """Current policy with the two metadata shortcuts disabled."""

    def update_scores(self, past_kv, attn_weights):
        super().update_scores(past_kv, attn_weights)
        self._stt_attn_idx = {}

    def _decode_step(self, past_kv):
        for layer, cache in self.layer_caches.items():
            def legacy(indices, cache=cache):
                promote, deferred = cache.resolve_promotions([(i, i + 1) for i in indices])
                return [s for s, _ in promote], [s for s, _ in deferred]

            cache.resolve_promotion_indices = legacy
        try:
            return super()._decode_step(past_kv)
        finally:
            for cache in self.layer_caches.values():
                # Restore class dispatch. Storing the original bound method
                # on its own cache instance creates a reference cycle that
                # retains GPU tensors across benchmark trials.
                del cache.resolve_promotion_indices


def policy(cls):
    return cls(sram_budget=1024, stt_budget=2048, start_size=4,
               recent_size=128, sttram_bifurcation_frac=.75,
               kv_dtype=torch.bfloat16)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True);ap.add_argument("--prompt-file",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--ctx",type=int,default=16384)
    ap.add_argument("--decode",type=int,default=16);ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--blocks",type=int,default=3)
    args=ap.parse_args()
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,device_map=args.device,
        attn_implementation="eager").eval()
    tok=AutoTokenizer.from_pretrained(args.model)
    prompt=Path(args.prompt_file).read_text()
    profile_one(model,tok,policy(TieredKVPolicy),args.ctx,args.decode,0,1,args.device,prompt)
    rows=[]
    for block in range(args.blocks):
        for name,cls in (("legacy",LegacyMetadataPolicy),("fast",TieredKVPolicy),
                         ("fast",TieredKVPolicy),("legacy",LegacyMetadataPolicy)):
            gc.collect()
            stats=profile_one(model,tok,policy(cls),args.ctx,args.decode,0,1,args.device,prompt)
            row={"block":block,"arm":name,"itl_ms":stats["itl_ms"],
                 "mean_itl_ms":stats["mean_itl_ms"],
                 "forward_ms":stats["forward_ms"],"ttft_ms":stats["ttft_ms"],
                 "decode_peak_gb":stats["peak_memory_gb"],
                 "step_ms":stats["trials"][0]["step_ms"]}
            rows.append(row)
            print({k:v for k,v in row.items() if k!="step_ms"},flush=True)
    summary={}
    for name in ("legacy","fast"):
        selected=[r for r in rows if r["arm"]==name]
        steps=[v for r in selected for v in r["step_ms"]]
        summary[name]={"median_itl_ms":statistics.median(steps),
                       "p95_itl_ms":float(np.quantile(steps,.95)),
                       "median_forward_ms":statistics.median(r["forward_ms"] for r in selected),
                       "max_decode_peak_gb":max(r["decode_peak_gb"] for r in selected)}
    paired=[]
    for block in range(args.blocks):
        old=statistics.median(r["itl_ms"] for r in rows if r["block"]==block and r["arm"]=="legacy")
        new=statistics.median(r["itl_ms"] for r in rows if r["block"]==block and r["arm"]=="fast")
        paired.append({"block":block,"legacy_ms":old,"fast_ms":new,
                       "improvement_pct":100*(old-new)/old})
    out={"scope":"same-process three-block ABBA, fixed real-text 16k prompt and decode length",
         "config":vars(args),"runtime":runtime_identity(),
         "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
         "rows":rows,"summary":summary,"paired_blocks":paired,
         "limitations":"One model, prompt, machine and seed; no quality or throughput claim."}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"summary":summary,"paired_blocks":paired},indent=2))


if __name__=="__main__": main()
