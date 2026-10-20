"""Read-only archive/AST/exact native-query projection-reuse proof."""
import ast,hashlib,inspect,json,sys,tarfile,textwrap
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from pool_query_selector import PoolQuerySelector
from reused_query_projection import PROJECTION_AST
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'plan.json');frozen=read(BASE/'implementation.json');raw=read(BASE/'gpu.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']
    assert sha(Path(plan['mistral_attention_source']))==plan['mistral_attention_source_sha256']
    for file,sources in [('control_source.tar.gz',plan['source_sha256']),('implementation_source.tar.gz',frozen['sources'])]:
        assert sha(BASE/file)==(plan['archive_sha256'] if file.startswith('control') else frozen['archive_sha256'])
        with tarfile.open(BASE/file) as t:
            archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
        assert archived==sources and all(sha(ROOT/p)==v for p,v in sources.items())
    for file,field,count in [('tests_focused.log','tests_focused_sha256',7),('tests_full.log','tests_full_sha256',273)]:
        assert sha(BASE/file)==frozen[field] and f'{count} passed' in (BASE/file).read_text()
    tree=ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    control=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='module.q_proj'];assert len(nodes)==1
    class Expected(ast.NodeTransformer):
        def visit_Call(self,node):
            if node is nodes[0]:return ast.Subscript(value=ast.Name(id='options',ctx=ast.Load()),
                slice=ast.Constant(value='_captured_query_projection'),ctx=ast.Load())
            return self.generic_visit(node)
    candidate=hashlib.sha256(ast.dump(Expected().visit(tree),include_attributes=False).encode()).hexdigest()
    assert PROJECTION_AST==raw['projection_ast']==dict(control=control,candidate=candidate)
    job=read(BASE/'gpu_job.json');assert job['status']=='finished' and job['exit_code']==0
    assert raw['passed'] and raw['hooks_canonical_method_restored'] and (raw['context'],raw['decode'])==(4096,12)
    a,b=raw['arms'];assert (a['arm'],b['arm'])==('cpu_control','cpu_reused')
    for field in ('predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes','host_initial_write_bytes'):assert a[field]==b[field],field
    assert len(a['frames'])==13 and len(a['selections'])==384 and len(a['predicted_token_ids'])==12
    assert a['q_projection_calls']==768 and b['q_projection_calls']==b['projection_reuses']==b['selected_layers']==384
    assert a['canonical_reads_during_selector']==b['canonical_reads_during_selector']==0 and b['pending_context_empty']
    older=read(ROOT/'results/hardware/pool_query_selector_v1/gpu_v2.json')
    old=next(r for r in older['arms'] if r['arm']=='cpu_paired')
    for field in ('predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes','host_initial_write_bytes'):assert a[field]==old[field],('frozen native-query replay',field)
    assert "args.cpu_batched_scores and args.strategy != 'lru'" in (ROOT/'experiments/evaluate_query_selection.py').read_text()
    output=BASE/'verified_correctness.json';assert not output.exists()
    result=dict(verified=True,tests_passed=273,focused_tests_passed=7,
        q_projection_calls_control=768,q_projection_calls_reused=384,
        exact_native_query_prediction_steps=12,exact_cache_logit_counter_frames=13,exact_visible_kv_choice_observations=384,
        exact_frozen_native_query_replay=True,native_h2d_bytes=b['host_read_bytes'],native_d2h_bytes=b['host_write_bytes'],
        canonical_cpu_reads_during_selector=0,pending_context_empty=True,own_hooks_canonical_method_restored=True,
        duplicate_projection_only_ast_change=True,ordinary_source_cli_defaults_unchanged=True,
        ready_for_visible_only_cost_diagnostic=True,fullscan_quality_timing_expansion=False,ordinary_adoption=False,full_goal_complete=False,
        limits='One native model/seed4K12 BF16 exact correctness; partial-error/context cleanup covered by focusedCPU tests. '
            'Reusing the projection does not remove cold full-scan traffic, rotary/ranking/hostID/concat overhead. '
            'No independent quality, resource match, latency win, full512MiB or physicalSTT proof.',
        evidence_sha256={f:sha(BASE/f) for f in ('plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
            'gpu.json','gpu_job.json','gpu.log','tests_focused.log','tests_full.log','verify_correctness.py')})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
