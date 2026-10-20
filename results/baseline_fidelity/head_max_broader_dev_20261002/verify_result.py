"""Audit reused development scores, control replay, source and ABBA prefill."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT/'src'))
import longbench_eval as bench
from provenance import read_jsonl
from report_remediation import compare
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = json.loads((BASE/'result.json').read_text())
    plan = json.loads((BASE/'plan.json').read_text())
    manifest = json.loads((BASE/'generation_manifest.json').read_text())
    needles = json.loads((BASE/'needles_manifest.json').read_text())
    assert raw['passed'] and len(raw['rows'])==24 and len(raw['prefill_rows'])==24
    for name,key in (('plan.json','plan_sha256'),('generation_manifest.json','manifest_sha256'),
                     ('needles_manifest.json','needles_sha256')):
        assert sha(BASE/name)==raw[key]
    frozen = json.loads((BASE/'source_v2_manifest.json').read_text())['source_sha256']
    with tarfile.open(BASE/'source_v2.tar.gz') as archive:
        for name,digest in raw['source_sha256'].items():
            assert digest == frozen[name]
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
            assert sha(ROOT/name)==digest, name
        executed_driver = archive.extractfile('experiments/check_head_max_development.py').read().decode()
    with tarfile.open(BASE/'source.tar.gz') as archive:
        original_driver = archive.extractfile('experiments/check_head_max_development.py').read().decode()
    assert executed_driver.replace('json.loads(json.dumps(result[field]))==controls[sample][field]',
                                   'result[field]==controls[sample][field]') == original_driver
    prior_path = ROOT/plan['prior_control']
    assert sha(prior_path)==plan['prior_control_sha256']
    prior = json.loads(prior_path.read_text())
    assert prior['passed']
    old = {r['sample_id']:r for r in prior['rows'] if r['arm']=='bounded' and r['seed']==0}
    tokenizer = AutoTokenizer.from_pretrained(str((ROOT/manifest['model']).resolve()))
    metrics, details = {}, []
    for task,entry in manifest['tasks'].items():
        source = ROOT/entry['source_file']
        assert sha(source)==entry['source_sha256']
        data = read_jsonl(source)
        paired = dict(sum_heads=[],tieredkv=[])
        for index,sample in zip(entry['selected_zero_based_indices'],entry['selected_sample_ids']):
            item = data[index]
            assert hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()==sample
            prompt = bench.DATASET_TO_PROMPT[task].format(**item)
            if task not in bench.NO_CHAT_TASKS:
                prompt = f'[INST] {prompt} [/INST]'
            rows = [r for r in raw['rows'] if r['sample_id']==sample]
            assert len(rows)==2 and {r['arm'] for r in rows}=={'sum_heads','max_head'}
            pair = {r['arm']:r for r in rows}
            for arm,row in pair.items():
                assert row['task']==task and row['seed']==0
                assert row['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
                assert row['max_gen']==bench.DATASET_TO_MAXGEN[task]
                ids = row['generated_token_ids']
                assert row['generation']['generated_tokens']==len(ids)
                assert tokenizer.decode(ids,skip_special_tokens=True)==row['prediction']
                assert 0 < len(ids) <= row['max_gen']
                assert tokenizer.eos_token_id not in ids[:-1]
                assert row['termination']==('eos' if ids[-1]==tokenizer.eos_token_id else 'cap')
                if row['termination']=='cap':
                    assert len(ids)==row['max_gen']
                expected = 100*max(bench.DATASET_TO_METRIC[task](row['prediction'],answer,
                    all_classes=item.get('all_classes',[])) for answer in item['answers'])
                assert abs(expected-row['score'])<1e-8
                assert row['reserved_host_tensor_bytes']==151257088
                assert len(row['prefill_query_shapes'])==32
                assert all(shape==[32,128,128] for shape in row['prefill_query_shapes'].values())
                if arm=='sum_heads':
                    for field in ('prediction','generated_token_ids','score','generation','max_gen','termination',
                        'prompt_sha256','state_sha256','trajectory_sha256','trajectory_frames','counters',
                        'prefill_query_shapes','explicit_h2d_payload_bytes','explicit_d2h_payload_bytes',
                        'occupied_host_tensor_bytes'):
                        assert row[field]==old[sample][field],(sample,field)
                paired['sum_heads' if arm=='sum_heads' else 'tieredkv'].append(row)
            details.append(dict(task=task,sample_id=sample,delta=pair['max_head']['score']-pair['sum_heads']['score'],
                scores={a:r['score'] for a,r in pair.items()},
                identical_output=pair['max_head']['generated_token_ids']==pair['sum_heads']['generated_token_ids']))
        metrics[task]={a:dict(n=len(rows),samples=rows,score=statistics.mean(r['score'] for r in rows))
                       for a,rows in paired.items()}
    quality = compare(metrics,'sum_heads',bootstrap=10000,seed=20261006,tasks=list(manifest['tasks']))
    quality_pass = quality['paired_bootstrap_95pct_interval'][0]>=-.5 and all(r['delta']>=-10 for r in details)
    timing = {}
    cases = {c['name']:c for c in needles['cases']}
    for name in plan['prefill_cases']:
        case = cases[name]
        assert hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest()==case['token_sha256']
        rows = [r for r in raw['prefill_rows'] if r['case']==name]
        assert len(rows)==12
        for block in range(3):
            block_rows=[r for r in rows if r['block']==block]
            assert [r['arm'] for r in block_rows]==['sum_heads','max_head','max_head','sum_heads']
        arms = {arm:[r for r in rows if r['arm']==arm] for arm in ('sum_heads','max_head')}
        for group in arms.values():
            assert len({r['placement_sha256'] for r in group})==1
        for row in rows:
            assert row['token_sha256']==case['token_sha256'] and row['prompt_tokens']==len(case['ids'])
            assert row['seconds']>0 and row['reserved_host_tensor_bytes']==151257088
            assert row['layer_counts']==rows[0]['layer_counts'] and len(row['layer_counts'])==32
            assert all(hot<=991 and cold<=1090 for hot,cold in row['layer_counts'].values())
        seconds = {a:statistics.median(r['seconds'] for r in group) for a,group in arms.items()}
        peaks = {a:max(r['peak_allocated_bytes'] for r in group) for a,group in arms.items()}
        change = 100*(seconds['max_head']/seconds['sum_heads']-1)
        timing[name]=dict(prompt_tokens=len(case['ids']),median_seconds=seconds,change_pct=change,
            peak_allocated_bytes=peaks,charged_prefill_guard_passed=change<=10,
            peak_guard_passed=peaks['max_head']<=peaks['sum_heads'])
    prefill_pass = all(r['charged_prefill_guard_passed'] and r['peak_guard_passed'] for r in timing.values())
    result = dict(verified=True, raw_sha256=sha(BASE/'result.json'),verifier_sha256=sha(Path(__file__)),
        source_archive_sha256=sha(BASE/'source_v2.tar.gz'),json_key_fix_only_verified=True,
        exact_old_controls=12,quality=quality,quality_gate_passed=quality_pass,details=details,
        regressions=sum(r['delta']<0 for r in details),losses_over_10=sum(r['delta']<-10 for r in details),
        prefill=timing,prefill_gate_passed=prefill_pass,
        advance_to_independent_preregistration=quality_pass and prefill_pass,
        limitation='Reused12developmentdocuments/sixfixedtasks/onegreedyseed; conditional bootstrap, not independent quality, physicalSTT, SnapKV, decode or throughput superiority. Importance changes initial cumulative scores and subsequent eviction as well as retained prompt positions.')
    output = BASE/'verified_report.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('details',)},indent=2))


if __name__=='__main__':
    main()
