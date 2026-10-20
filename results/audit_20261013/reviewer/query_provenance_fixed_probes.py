"""Mocked CPU runner integration: no model loading or GPU work."""
import hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
import compare_query_latency as runner
checks=[]
for flag in [False,True]:
    events=[]; forwarded=[]; stored=[]
    sentinel={'snapshot_path':'CPU_probe_only','artifact_sha256':{'model':'frozen'}}
    def identity(location):
        events.append('identity');return sentinel
    model=SimpleNamespace();model.eval=lambda:model
    def load(*a,**kw):
        events.append('model_load');return model
    def profile(*a,**kw):
        forwarded.append(kw['prefill_last_logits']);return {'itl_ms':0.0}
    with tempfile.TemporaryDirectory(dir=ROOT/'results/audit_20261013/reviewer') as td:
        prompt=Path(td)/'prompt.txt';prompt.write_text('CPU fake prompt')
        argv=['compare','--model',td,'--prompt-file',str(prompt),'--output',str(Path(td)/'output.json')]
        if flag:argv.append('--prefill-last-logits')
        with patch.object(sys,'argv',argv),patch.object(runner,'model_artifact_identity',identity),\
             patch.object(runner.AutoModelForCausalLM,'from_pretrained',load),\
             patch.object(runner.AutoTokenizer,'from_pretrained',lambda *a,**k:object()),\
             patch.object(runner,'QuerySelector',lambda *a:SimpleNamespace(close=lambda:None)),\
             patch.object(runner,'profile_one',profile),patch.object(runner,'runtime_identity',lambda:{}),\
             patch.object(runner,'atomic_json',lambda path,obj:stored.append(obj)):
            runner.main()
    assert events==['identity','model_load']
    assert forwarded==[flag]*4
    assert stored[0]['config']['model_identity']==sentinel
    assert stored[0]['config']['prefill_last_logits']==flag
    checks.append({'last_logits':flag,'forwarded_four_arms':forwarded,'identity_precedes_model_loading':True,'identity_serialized':True})
print(json.dumps({'cpu_mock_only':True,'all_checks_passed':True,'checks':checks,
 'source_sha256':hashlib.sha256((ROOT/'experiments/compare_query_latency.py').read_bytes()).hexdigest()},indent=2))
