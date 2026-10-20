"""Account for initial cold placement as well as decoder writes; no device claim."""
from pathlib import Path
import json,hashlib
from collections import defaultdict
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
def counters(initial,paid,saved):
 assert min(initial,paid,saved)>=0
 den=initial+paid+saved
 return {'initial_cold_payload_row_writes':initial,'decoder_paid_writes':paid,'decoder_avoided_writes':saved,'whole_request_paid_payload_row_writes':initial+paid,'potential_row_writes_without_backup_reuse':den,'whole_request_avoided_percent':100*saved/den if den else None,'decoder_only_avoided_percent':100*saved/(paid+saved) if paid+saved else None,'whole_request_paid_payload_bytes':(initial+paid)*4096}
def main():
 proof=json.loads((BASE/'verified_results.json').read_text());assert proof['verified']
 output=BASE/'whole_request_writes.json';assert not output.exists()
 sums=defaultdict(lambda:[0,0,0]);hashes=proof['raw_record_sha256']
 for name,digest in hashes.items():
  p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  r=json.loads(p.read_text());arm=r.get('arm','frac075')
  if arm=='tiered_hot1024':arm='frac075'
  initial=0
  for layer in r['geometry_after_prefill_before_first_decode']['layers']:
   tensors=layer['tier_payloads'];shape=tensors['stt_k']['shape'];assert shape[0]==8 and shape[2]==128
   assert tensors['stt_k']['logical_bytes']==tensors['stt_v']['logical_bytes']==shape[1]*8*128*2
   initial+=shape[1]
  cs=r['cache_stats'];assert cs['paid_writes']+cs['writes_saved']==cs['demoted']
  for population in ['all_tasks']+(['five_qa'] if r['task']!='gov_report' else []):
   total=sums[population,arm]
   for i,n in enumerate((initial,cs['paid_writes'],cs['writes_saved'])):total[i]+=n
 report={'verified':True,'definition_frozen_in_script_before_ablation_outcomes':True,'ablation_proof_sha256':hashlib.sha256((BASE/'verified_results.json').read_bytes()).hexdigest(),'records':len(hashes),'scope':'Logical STT KV payload writes: initial retained cold placement once plus paid decoder demotions. No metadata, controller, interface, VRAM prefill writes, or physical endurance measurement. All methods free-generate; output lengths differ. No inclusive-vs-exclusive causal ablation is supplied by fraction comparisons.','evidence':['src/tiered_kv_cache.py initial_bifurcation copies stt_k/stt_v once','experiments/longbench_eval.py decoder StepRecord counts demotions only'],'populations':{pop:{a:counters(*values) for (p,a),values in sums.items() if p==pop} for pop in ['all_tasks','five_qa']}}
 output.write_text(json.dumps(report,indent=2)+'\n');print('Whole-request logical write accounting verified',len(hashes))
if __name__=='__main__':
 assert counters(10,0,90)['whole_request_avoided_percent']==90
 assert counters(0,0,0)['whole_request_avoided_percent'] is None
 assert counters(10,0,0)['whole_request_paid_payload_bytes']==40960
 main()
