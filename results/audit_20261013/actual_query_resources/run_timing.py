"""Reuse the existing ABBA profiler with quiet guards and a frozen input stream."""
from pathlib import Path
import hashlib, importlib.util, json, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments'))
import compare_query_latency as driver
from provenance import atomic_json
BASE = Path(__file__).resolve().parent
plan = json.loads((BASE/'plan.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (BASE/'timing.json').exists() and not (BASE/'observations.json').exists()
helper = ROOT/plan['quiet_helper']
assert sha(helper) == plan['quiet_helper_sha256']
spec = importlib.util.spec_from_file_location('quiet_observer', helper)
observer = importlib.util.module_from_spec(spec); spec.loader.exec_module(observer)
stream_path = ROOT/plan['teacher_stream']
assert sha(stream_path) == plan['teacher_stream_sha256']
stream = json.loads(stream_path.read_text())['config']['teacher_token_ids']
report = {'passed': False, 'rows': [], 'quiet_observations': [], 'plan_sha256': sha(BASE/'plan.json')}
original = driver.profile_one

def check(stage):
    item = dict(stage=stage, **observer.quiet())
    report['quiet_observations'].append(item)
    atomic_json(BASE/'observations.json', report)
    assert item['quiet'], 'Unexpected GPU0 compute process; preserve partial result'

def measured(*args, **options):
    check('before_row')
    options['forced_token_ids'] = stream
    result = original(*args, **options)
    check('after_row')
    report['rows'].append(result)
    atomic_json(BASE/'observations.json', report)
    return result

driver.profile_one = measured
sys.argv = ['timing', '--model', '../models/mistral-7b-instruct',
    '--prompt-file', plan['prompt'], '--output', str(BASE/'timing.json'),
    '--compare-prefill', '--prefill-strategy', 'lru', '--blocks', '3', '--prefill-last-logits']
try:
    check('before_model_load')
    driver.main()
    report['passed'] = True
finally:
    atomic_json(BASE/'observations.json', report)
