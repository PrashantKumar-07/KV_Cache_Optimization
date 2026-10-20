"""Freeze the complete six-file evaluation before observing new predictions."""
from pathlib import Path
import hashlib,json,sys,tarfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import longbench_eval as bench
from provenance import atomic_json,model_artifact_identity,runtime_identity,read_jsonl
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'plan.json').exists(), 'Frozen plan may not be overwritten'
old=json.loads((ROOT/'results/audit_20261015_query_combination/plan.json').read_text())
tasks=['multifieldqa_en','qasper','narrativeqa','hotpotqa','triviaqa','gov_report']
files={t:old['dataset_files'][t] for t in tasks}
rows={t:read_jsonl(ROOT/p) for t,p in files.items()}
ids={t:[hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() for r in rr] for t,rr in rows.items()}
source={str(p.relative_to(ROOT)):sha(p) for folder in ('src','experiments','tests','sota') for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.cpp','.cu','.h','.hpp','.sh','.json')}
arms=['full','streamingllm','h2o_adapted','snapkv_adapted','tiered_hot1024','tiered_hot992','snapkv_faithful']
plan={
 'schema':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'Fresh complete original-source six-task evaluation; retain original figure questions and styles. No SOTA success assumed.',
 'model':old['model'],'model_identity':model_artifact_identity(old['model']),
 'source_sha256':source,'runtime':runtime_identity(),
 'runner_sha256':sha(BASE/'run_full.py'),'preparation_sha256':sha(Path(__file__)),
 'tasks':tasks,'dataset_files':files,'dataset_sha256':{t:sha(ROOT/p) for t,p in files.items()},
 'sample_ids':ids,'counts':{t:len(rr) for t,rr in rows.items()},'total_rows':sum(map(len,rows.values())),
 'arms':arms,'shards':2,'total_predictions':sum(map(len,rows.values()))*len(arms),
 'protocol':{'seed':0,'dtype':'bfloat16','max_ctx':31500,'batch_size':1,'decoding':'greedy; stop on tokenizer EOS or fixed task limit','max_gen':{t:bench.DATASET_TO_MAXGEN[t] for t in tasks},'prefill_backend':'sdpa','decode_backend':'eager','last_logits_only':True,'position_ids':'absolute contiguous positions following middle split','truncation':'first15750+last15750 tokens if longer than31500','prompt_templates':{t:bench.DATASET_TO_PROMPT[t] for t in tasks},'wrapping':'[INST] prompt [/INST] except original NO_CHAT_TASKS','no_chat_tasks':list(bench.NO_CHAT_TASKS),'metrics':{t:bench.DATASET_TO_METRIC[t].__name__ for t in tasks},'trivia_postprocessing':'strip leading newlines; first line','source_mutations_during_run':False},
 'arm_configs':{
  'full':{'description':'uncompressed reference, unrestricted KV; not matched budget'},
  'streamingllm':{'description':'adapted absolute-position sink/window control','sink':4,'recent':1020},
  'h2o_adapted':{'description':'repository adapted K-as-Q initialization, shared retained rows','budget':1024,'sink':4,'recent':128},
  'snapkv_adapted':{'description':'repository proxy prefill selector plus H2O decode; not faithful upstream SnapKV','budget':1024,'sink':4,'recent':128,'pool_kernel':5},
  'tiered_hot1024':{'hot':1024,'cold':2048,'quota':32,'sink':4,'recent':128,'frac':.75,'prompt_queries':'actual','decode_selection':'existing recency','storage':'ordinary GPU algorithmic simulation; not native deployment timing'},
  'tiered_hot992':{'hot':992,'cold':2048,'quota':32,'sink':4,'recent':128,'frac':.75,'prompt_queries':'actual','decode_selection':'existing recency','storage':'ordinary GPU algorithmic simulation; not native deployment timing'},
  'snapkv_faithful':{'description':'upstream-selector faithful Mistral adapter, expanded32KV heads, decode cache grows; separate fidelity comparison, not equal physical memory','capacity':1024,'window':32,'reuse_projection':True}},
 'analysis':{'task_summary':'arithmetic mean per-example score times100; report each task and equal-task macro mean','paired_comparisons':'all rows paired by task/sample; no selective deletion','uncertainty':'target-document-cluster bootstrap,10000 draws,seed0; preserve paired arms; report task and equal-task macro95% intervals','historical135':'first25 rows of first5tasks; first10 gov_report; separately labeled reproduction stratum','population_scope':'full1474 is descriptive and includes consumed data; document-disjoint unconsumed stratum reported separately; narrative has no unconsumed documents','candidate_lock':'actual-query prefill/recency candidate fixed before this run; no tuning, configurations and analysis unchanged after unblinding','failed_candidate':'current-query combination excluded after registered screen failure; do not resurrect by expanding sample','resource_scope':'1024-hot and992+32-attended panels reuse same freshly generated control records. Label configured vs observed width, full-cache unrestricted, faithfulSnap unequal expanded/growing resources. No full-memory equality claim.','timing_scope':'concurrent GPU evaluation walltimes are diagnostics only; do not use as quiet latency benchmark','figures':'regenerate only complete verified corresponding experiments; quality run does not validate distinct context-scaling/headroom ablations'},
 'partition_sha256':sha(BASE/'partitions.json'),
 'resumption':'two deterministic row shards (global index modulo2); atomic individual arm records, validate plan hash and identity; preserve failed attempts, never silently omit failed rows; no favorable reruns'}
plan['analysis']['primary_contrast']={'population':'unconsumed_as_of_inventory five-task stratum; no NarrativeQA','candidate':'tiered_hot992','control':'snapkv_faithful','endpoint':'equal-task macro mean score difference','interval':'paired target-document cluster bootstrap,10000 draws,seed0,95% percentile interval','interpretation':'one preregistered quality contrast under disclosed unequal cache geometry, not a SOTA or equal-physical-memory test'}
plan['analysis']['secondary_contrasts']='All other method/task/population comparisons are exploratory, with unadjusted95% intervals labeled descriptive; no choosing significant subset as primary and no formal multiple-comparison superiority claim.'
plan['configuration_authority']='Exact frozen runner hash binds executable policy constructors; arm_configs describes them. No command-line policy overrides.'
assert plan['total_rows']==1474 and plan['total_predictions']==10318
atomic_json(BASE/'plan.json',plan)
with tarfile.open(BASE/'source.tar.gz','x:gz') as out:
 for name in source:out.add(ROOT/name,arcname=name)
atomic_json(BASE/'freeze.json',{'plan_sha256':sha(BASE/'plan.json'),'source_archive_sha256':sha(BASE/'source.tar.gz'),'files':len(source),'total_predictions':plan['total_predictions']})
print(json.dumps({'counts':plan['counts'],'total_predictions':plan['total_predictions'],'plan_sha256':sha(BASE/'plan.json')},indent=2))
