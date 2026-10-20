"""Independent verification of a saved prefix; never a final result aggregate."""
import ast,hashlib,json,re,string,sys
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
from rouge import Rouge
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments'))
from provenance import read_jsonl,atomic_json
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text());ph=sha(BASE/'plan.json')
assert ph==json.loads((BASE/'freeze.json').read_text())['plan_sha256']
assert sha(BASE/'run_full.py')==plan['runner_sha256']
assert sha(BASE/'partitions.json')==plan['partition_sha256']
assert all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items())
reference=ROOT/'sota/snapkv/experiments/LongBench/metrics.py'
names={'normalize_answer','f1_score','qa_f1_score','rouge_score'}
functions=[n for n in ast.parse(reference.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in functions}==names
ns=dict(re=re,string=string,Counter=Counter,Rouge=Rouge)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(reference),'exec'),ns)
tokenizer=AutoTokenizer.from_pretrained(plan['model'])
data={t:read_jsonl(ROOT/p) for t,p in plan['dataset_files'].items()}
paths=sorted(BASE.glob('records/shard_*/*.json'))
seen=set();pairs=defaultdict(list);counts=Counter();checked=[]
for path in paths:
 r=json.loads(path.read_text())
 if 'arm' not in r:continue
 assert r['execution_complete'] and r['plan_sha256']==ph
 t,i,a=r['task'],r['physical_file_row_index'],r['arm'];key=(t,i,a)
 assert key not in seen;seen.add(key)
 item=data[t][i];sid=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
 assert sid==r['sample_id']==plan['sample_ids'][t][i]
 prompt=plan['protocol']['prompt_templates'][t].format(**item)
 if t not in plan['protocol']['no_chat_tasks']:prompt='[INST] '+prompt+' [/INST]'
 ids=tokenizer(prompt,return_tensors='np').input_ids
 if ids.shape[1]>31500:
  import numpy as np
  ids=np.concatenate((ids[:,:15750],ids[:,-15750:]),axis=1)
 assert r['input_ids_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
 assert r['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
 assert tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True)==r['prediction']
 assert 1<=len(r['generated_token_ids'])<=plan['protocol']['max_gen'][t]
 assert tokenizer.eos_token_id not in r['generated_token_ids'][:-1]
 assert len(r['generated_token_ids'])==plan['protocol']['max_gen'][t] or r['generated_token_ids'][-1]==tokenizer.eos_token_id
 text=r['prediction'].lstrip('\n').split('\n')[0] if t=='triviaqa' else r['prediction']
 fn=ns['rouge_score' if t=='gov_report' else 'qa_f1_score']
 score=100*max(fn(text,x) for x in item['answers'])
 assert abs(score-r['score'])<1e-10
 assert r['all_model_logits_finite'] and r['hooks_and_groups_restored']
 assert r['decode_attended_width']['layer_calls']==32*(len(r['generated_token_ids'])-1)
 assert r['geometry_after_prefill_before_first_decode']['layers'][0]['visible_k']['dtype']=='torch.bfloat16'
 pairs[t,i].append(r);counts[a]+=1
 checked.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
for rows in pairs.values():
 assert len({r['input_ids_sha256'] for r in rows})==1
 assert len({r['generated_token_ids'][0] for r in rows})==1
assert checked, 'No real predictions checked'
out=BASE/'startup_verification.json';assert not out.exists()
atomic_json(out,{'verified_utc':datetime.now(timezone.utc).isoformat(),'verified':True,'scope':'Snapshot prefix integrity/rescoring only; no final aggregates or quality conclusion','plan_sha256':ph,'records_verified':len(checked),'counts_by_arm':dict(counts),'unique_rows':len(pairs),'complete_seven_arm_rows':sum(len(rr)==7 for rr in pairs.values()),'independent_metric_reference':str(reference.relative_to(ROOT)),'reference_sha256':sha(reference),'checks':'source/plan/runner/partition/sample/prompt/inputtoken/decode/EOS/length/reference-score/finite/cleanup/layercall/firsttoken/dtype','records':checked,'verifier_sha256':sha(Path(__file__))})
print(json.dumps({'verified':True,'records':len(checked),'paired_rows':len(pairs),'scope':'prefix integrity only; full run still required'}))
