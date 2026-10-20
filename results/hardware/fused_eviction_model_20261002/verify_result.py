"""Independently verify exact combined-policy parity and full-model speed gates."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(a,b,frames):
    for field in ('state_sha256','history_sha256','every_visible_state_sha256',
        'explicit_h2d_payload_bytes','explicit_d2h_payload_bytes','initial_d2h_payload_bytes',
        'occupied_host_bytes','reserved_host_bytes','native_d2h_calls','native_h2d_calls','per_layer_initial_d2h_calls'):
        assert a[field]==b[field],field
    assert len(a['every_visible_state_sha256'])==len(b['every_visible_state_sha256'])==frames
    assert a['reserved_host_bytes']==402784256
    assert len(a['trials'])==len(b['trials'])==1
    for field in ('predicted_token_ids','paid_writes','writes_saved','demoted','promoted','decode_steps'):
        assert a['trials'][0][field]==b['trials'][0][field],field


def main():
    plan=json.loads((BASE/'plan.json').read_text())
    hashes=json.loads((BASE/'source_manifest.json').read_text())['source_sha256']
    with tarfile.open(BASE/'source.tar.gz') as archive:
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest,name
        core=archive.extractfile('src/tiered_kv_cache.py').read()
    for name,digest in plan['sources'].items():
        assert sha(ROOT/name)==digest
    screen_path=ROOT/plan['screen']
    assert sha(screen_path)==plan['screen_sha256']
    screen=json.loads(screen_path.read_text())
    assert screen['verified'] and screen['advance_to_model'] and screen['median_saving_ms']>=4
    binary=Path('/data/nishant/tieredkv_external/fused_hot_compact_build_20261002/tieredkv_hot_compact_v1.so')
    assert sha(binary)==screen['binary_sha256']
    # Audit that the compiled method keeps every original noncompaction node.
    sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
    from fused_eviction_policy import CONTROL_AST,DEFERRED_AST
    original=ast.parse(core)
    cls=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='TieredKVCache')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evict_and_demote')
    tree=ast.Module(body=[method],type_ignores=[])
    assert ast.dump(tree)==CONTROL_AST
    node=next(n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='victims')
    node.body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),
        attr='_compact_or_queue',ctx=ast.Load()),args=[ast.Name(id='victims',ctx=ast.Load())],keywords=[]))]
    assert ast.dump(tree)==DEFERRED_AST
    assert sha(ROOT/'experiments/fused_eviction_policy.py')==hashes['experiments/fused_eviction_policy.py']
    synthetic=json.loads((BASE/'synthetic.json').read_text())
    assert synthetic['passed'] and synthetic['fused'] and len(synthetic['checks'])==32
    assert synthetic['binary_sha256']==screen['binary_sha256']
    for name,digest in synthetic['source_sha256'].items():
        assert hashes[name]==digest
    assert all(r['exact_state_payload_counters_and_cleanup'] and r['appended_tokens']==18 and
        r['decode_calls']==12 and r['last_query_attention_summary'] for r in synthetic['checks'])
    fused=sum(r['fused_compactions'] for r in synthetic['checks'])
    eager=sum(r['eager_compactions'] for r in synthetic['checks'])
    assert fused>0 and eager>0
    parity=json.loads((BASE/'parity.json').read_text())
    abba=json.loads((BASE/'abba.json').read_text())
    teacher=json.loads((ROOT/'results/shadow_ablation_full/fixed_stream_20260929.json').read_text())['config']['teacher_token_ids']
    for raw,decode in ((parity,12),(abba,50)):
        assert raw['passed'] and raw['config']['decode']==decode and raw['config']['real_query_prefill']
        assert raw['runner_sha256']==hashes['experiments/compare_cpu_batched.py']
        assert raw['fused_binary_sha256']==screen['binary_sha256']
        assert raw['teacher_token_ids']==teacher[:decode+1]
        for name,digest in raw['fused_source_sha256'].items():
            assert hashes[name]==digest
        for key,name in (('backend_sha256','cpu_batched_cache'),('slot_backend_sha256','cpu_slot_cache'),
            ('query_capture_sha256','evaluate_query_selection'),('score_backend_sha256','batched_score_policy')):
            assert raw[key]==hashes[f'experiments/{name}.py']
    assert len(parity['rows'])==8 and parity['config']['blocks']==0 and parity['config']['warmup']==0
    assert parity['prompt_sha256']==plan['sources']['results/hardware/batched_eviction_coordinator_20261001/max_context_prompt.txt']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str((ROOT/plan['model']).resolve()))
    old=tokenizer((ROOT/'results/remediation_v3/latency_prompt.txt').read_text(),return_tensors='pt').input_ids
    extended=tokenizer((ROOT/'results/hardware/batched_eviction_coordinator_20261001/max_context_prompt.txt').read_text(),return_tensors='pt').input_ids
    assert extended.shape[1]>=31500
    for i,ctx in enumerate((4096,8192,16384,31500)):
        a,b=parity['rows'][2*i:2*i+2]
        assert a['context']==b['context']==ctx and a['arm']=='direct_scores' and b['arm']=='fused_scores'
        exact(a,b,13)
        assert b['fused_compactions']>0
        if ctx<31500:
            assert np.array_equal(old[:,:ctx].numpy(),extended[:,:ctx].numpy())
    assert len(abba['rows'])==12 and abba['config']['blocks']==3 and abba['config']['warmup']==1
    assert abba['prompt_sha256']==plan['sources']['results/remediation_v3/latency_prompt.txt']
    rows=abba['rows'];control,candidate='direct_scores','fused_scores'
    for block in range(3):
        group=rows[4*block:4*block+4]
        assert [r['arm'] for r in group]==[control,candidate,candidate,control]
        assert all(r['block']==block and r['context']==16384 for r in group)
    for row in rows[1:]:
        exact(rows[0],row,0)
    summary={}
    for arm in (control,candidate):
        chosen=[r for r in rows if r['arm']==arm]
        steps=[v for r in chosen for v in r['trials'][0]['step_ms']]
        assert len(steps)==300 and all(v>0 for v in steps)
        summary[arm]=dict(median_ms=statistics.median(steps),p95_ms=float(np.percentile(steps,95)),
            ttft_ms=statistics.median(r['trials'][0]['ttft_ms'] for r in chosen),
            prefill_peak_gb=max(r['prefill_peak_memory_gb'] for r in chosen),
            decode_peak_gb=max(r['peak_memory_gb'] for r in chosen))
    blocks=[{arm:statistics.median(v for r in rows if r['block']==block and r['arm']==arm for v in r['trials'][0]['step_ms']) for arm in (control,candidate)} for block in range(3)]
    a,b=summary[control],summary[candidate]
    gate=(b['median_ms']<=.95*a['median_ms'] and b['p95_ms']<=1.02*a['p95_ms'] and
        all(row[candidate]<row[control] for row in blocks))
    assert gate==abba['summary']['16384']['latency_gate_passed']
    guards=dict(ttft=b['ttft_ms']<=1.10*a['ttft_ms'],prefill_peak=b['prefill_peak_gb']<=1.02*a['prefill_peak_gb'])
    report=dict(verified=True,source_archive_sha256=sha(BASE/'source.tar.gz'),binary_sha256=screen['binary_sha256'],
        whole_original_eviction_except_compaction_ast_verified=True,synthetic_cases=32,
        synthetic_fused_compactions=fused,synthetic_eager_compactions=eager,model_contexts=[4096,8192,16384,31500],
        boundary_frames_per_context_arm=13,summary=summary,blocks=blocks,
        median_improvement_pct=100*(1-b['median_ms']/a['median_ms']),latency_gate_passed=gate,
        regression_guards=guards,advance_to_mandatory_stress=gate and all(guards.values()),
        evidence_sha256={name:sha(BASE/name) for name in ('plan.json','synthetic.json','parity.json','abba.json')},
        verifier_sha256=sha(Path(__file__)),
        limitations='One model/seed/fixedteacherstream, same CPU-DRAM backend and unchanged logical/quality policy. Coordinator+fused compaction measured together; no attribution to kernel alone, SnapKV/quality/physicalSTT/resources/serving superiority, or ordinary CLI promotion without mandatory stress.')
    output=BASE/'verified_report.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
