"""Paired quality report for a bounded promotion-admission pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--control",required=True)
    ap.add_argument("--candidate",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    a=json.loads(Path(args.control).read_text())
    b=json.loads(Path(args.candidate).read_text())
    variable={"tiered_promotion_budget"}
    ac={k:v for k,v in a["config"].items() if k not in variable}
    bc={k:v for k,v in b["config"].items() if k not in variable}
    if ac!=bc: raise ValueError("non-budget config/source mismatch")
    if a["config"]["tiered_promotion_budget"]==b["config"]["tiered_promotion_budget"]:
        raise ValueError("promotion budgets are identical")
    if set(a["results"])!=set(b["results"]): raise ValueError("task mismatch")

    rng=np.random.default_rng(0);task_draws=[];tasks={}
    for task in a["results"]:
        control=a["results"][task]["tieredkv"]
        candidate=b["results"][task]["tieredkv"]
        x,y=control["samples"],candidate["samples"]
        if len(x)!=len(y) or [z["sample_id"] for z in x]!=[z["sample_id"] for z in y]:
            raise ValueError(f"sample mismatch: {task}")
        diff=np.array([v["score"]-u["score"] for u,v in zip(x,y)])
        task_draws.append(rng.choice(diff,(10000,len(diff)),replace=True).mean(1))
        tasks[task]={"n":len(diff),"control":control["score"],
                     "candidate":candidate["score"],"difference":float(diff.mean()),
                     "identical_predictions":sum(u["prediction"]==v["prediction"] for u,v in zip(x,y))}
    draws=np.stack(task_draws).mean(0)
    report={"scope":"paired budget pilot; same source/runtime/model/data/samples/config except promotion admission",
            "control_budget":a["config"]["tiered_promotion_budget"],
            "candidate_budget":b["config"]["tiered_promotion_budget"],
            "tasks":tasks,
            "task_balanced_difference":float(np.mean([v["difference"] for v in tasks.values()])),
            "paired_bootstrap_95pct_interval":np.quantile(draws,[.025,.975]).tolist(),
            "limitations":"Selected task/sample pilot; interval excludes model/seed/task uncertainty."}
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
