"""Fail-fast 16K sweep separating slow-tier exposure from VRAM admission."""
import argparse, hashlib, json, statistics, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/p) for p in ("src","experiments")]
from latency_profile import profile_one
from longbench_eval import TieredKVPolicy
from provenance import runtime_identity

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True);ap.add_argument("--prompt-file",required=True)
    ap.add_argument("--output",required=True);ap.add_argument("--ctx",type=int,default=16384)
    ap.add_argument("--decode",type=int,default=32);ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--budgets",nargs="+",type=int,default=[32,0,2,4,8,16,32])
    args=ap.parse_args()
    if any(x<0 for x in args.budgets): raise ValueError("budgets must be nonnegative")
    text=Path(args.prompt_file).read_text()
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        device_map=args.device,attn_implementation="eager").eval()
    tok=AutoTokenizer.from_pretrained(args.model);rows=[]
    for budget in args.budgets:
        policy=TieredKVPolicy(sram_budget=1024,stt_budget=2048,start_size=4,
            recent_size=16,sttram_bifurcation_frac=.75,stt_expose_quota=32,
            promotion_budget=budget,kv_dtype=torch.bfloat16)
        stats=profile_one(model,tok,policy,args.ctx,args.decode,0,1,args.device,text)
        counters=policy.stats()
        row={"promotion_budget":budget,"itl_ms":stats["itl_ms"],
             "mean_itl_ms":stats["mean_itl_ms"],"forward_ms":stats["forward_ms"],
             "decode_peak_memory_gb":stats["peak_memory_gb"],
             "promoted":counters["promoted"],"demoted":counters["demoted"],
             "paid_writes":counters["paid_writes"],"writes_saved":counters["writes_saved"]}
        rows.append(row);print(row,flush=True)
    grouped={}
    for row in rows: grouped.setdefault(row["promotion_budget"],[]).append(row["itl_ms"])
    out={"scope":"one fixed real-text prompt; exposure fixed at 32; migration admission swept",
         "config":vars(args),"prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),
         "runtime":runtime_identity(),"rows":rows,
         "median_itl_ms":{str(k):statistics.median(v) for k,v in grouped.items()},
         "limitations":"Latency/counter screen only; no quality conclusion."}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":main()
