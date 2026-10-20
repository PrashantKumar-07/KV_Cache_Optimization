"""Scoped opt-in architecture pilot over the existing paired research runner."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments'))
import evaluate_query_selection as driver
from grouped_retention import GroupedRetentionPolicy
arm=sys.argv[1];assert arm in ('control','candidate')
config=json.loads((ROOT/'../models/mistral-7b-instruct/config.json').read_text())
assert config['model_type']=='mistral' and config.get('sliding_window') is None
original=driver.MeasuredTieredPolicy
class MeasuredGrouped(original,GroupedRetentionPolicy):pass
if arm=='candidate':driver.MeasuredTieredPolicy=MeasuredGrouped
resume=driver.resume_results
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def scoped_resume(path,settings):
 settings.update(head_grouped_retention=arm=='candidate',pilot_runner_sha256=sha(Path(__file__)),grouped_policy_sha256=sha(ROOT/'experiments/grouped_retention.py'))
 return resume(path,settings)
driver.resume_results=scoped_resume
sys.argv=['pilot','--model','../models/mistral-7b-instruct','--data-dir','../hf_cache/datasets/downloads/extracted/5def9c7eaf83951e1cee75902d8611ef764ad88dd289e8f0bf97ff658a3eb83a/data','--tasks','qasper','narrativeqa','multifieldqa_en','--max-samples','2','--strategy','lru','--real-query-prefill','--prefill-last-logits','--output',str(Path(__file__).with_name(arm+'.json'))]
try:driver.main()
finally:driver.MeasuredTieredPolicy=original;driver.resume_results=resume
