#!/usr/bin/env python3
"""Validate paired 2048-token bounded-prefill pilot across three tasks."""
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json


def main():
    root=Path('results/bounded_prefill')
    pairs=(('exact_5.json','chunk2048_5.json'),
           ('exact_more.json','chunk2048_more.json'))
    exact,chunked={},{}
    configs=[]
    for af,bf in pairs:
        a,b=(json.loads((root/f).read_text()) for f in (af,bf))
        configs.extend((a['config'],b['config']))
        exact.update(a['results']); chunked.update(b['results'])
    variable={'tasks','dataset_sha256','tiered_prefill_chunk_size'}
    frozen={k:v for k,v in configs[0].items() if k not in variable}
    assert all({k:v for k,v in c.items() if k not in variable}==frozen for c in configs)
    assert {configs[i]['tiered_prefill_chunk_size'] for i in range(0,len(configs),2)}=={0}
    assert {configs[i]['tiered_prefill_chunk_size'] for i in range(1,len(configs),2)}=={2048}
    assert set(exact)==set(chunked)=={'qasper','narrativeqa','gov_report'}
    rng=np.random.default_rng(0); task_draws=[]; tasks={}
    for task in exact:
        a,b=exact[task]['tieredkv'],chunked[task]['tieredkv']
        assert a['n']==b['n']==5
        assert [x['sample_id'] for x in a['samples']]==[x['sample_id'] for x in b['samples']]
        diff=np.array([y['score']-x['score'] for x,y in zip(a['samples'],b['samples'])])
        task_draws.append(rng.choice(diff,(10000,5),replace=True).mean(1))
        tasks[task]={'exact':a['score'],'chunked':b['score'],'difference':float(diff.mean()),
                     'identical_predictions':sum(x['prediction']==y['prediction'] for x,y in zip(a['samples'],b['samples']))}
    draws=np.stack(task_draws).mean(0)
    report={'scope':'Three selected tasks × first five examples; chunked minus exact; source/runtime/config matched',
            'tasks':tasks,'task_balanced_difference':float(np.mean([x['difference'] for x in tasks.values()])),
            'paired_bootstrap_95pct_interval':np.quantile(draws,[.025,.975]).tolist(),
            'limitations':'Conditional on 15 selected examples, one model/seed; no preservation or improvement claim.'}
    atomic_json(root/'quality_2048_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
