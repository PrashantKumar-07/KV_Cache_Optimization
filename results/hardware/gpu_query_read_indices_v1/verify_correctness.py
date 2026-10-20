"""Frozen CPU/native/model exactness proof for GPU native read-index maps."""
import ast,hashlib,inspect,json,sys,tarfile,textwrap
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];sys.path.insert(0,str(ROOT/'experiments'))
from pool_query_selector import PoolQuerySelector
from gpu_query_read_indices import READ_AST
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
    plan=read(BASE/'plan.json');frozen=read(BASE/'implementation.json');raw=read(BASE/'gpu.json');native=read(BASE/'native.json')
    assert sha(BASE/'plan.json')==frozen['plan_sha256']
    for file,sources in [('control_source.tar.gz',plan['source_sha256']),('implementation_source.tar.gz',frozen['sources'])]:
        assert sha(BASE/file)==(plan['archive_sha256'] if file.startswith('control') else frozen['archive_sha256'])
        with tarfile.open(BASE/file) as t:
            archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
        assert archived==sources and all(sha(ROOT/p)==v for p,v in sources.items())
    for file,field,count in [('tests_focused_v2.log','tests_focused_sha256',10),('tests_full.log','tests_full_sha256',283)]:
        assert sha(BASE/file)==frozen[field] and f'{count} passed' in (BASE/file).read_text()
    tree=ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    control=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest();changed=[]
    class Expected(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='live':
                assert ast.unparse(node.value)=='torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)'
                node.value=ast.parse('self.policy.query_reader.indices(index)',mode='eval').body;changed.append('live')
            return self.generic_visit(node)
        def visit_Call(self,node):
            if ast.unparse(node.func)=='cache.select_stt_pair':
                assert ast.unparse(node)=='cache.select_stt_pair(live.tolist())'
                changed.append('read');return ast.parse('self.policy.query_reader.read_live(index)',mode='eval').body
            return self.generic_visit(node)
    candidate=hashlib.sha256(ast.dump(Expected().visit(tree),include_attributes=False).encode()).hexdigest()
    assert sorted(changed)==['live','read'] and READ_AST==raw['read_ast']==dict(control=control,candidate=candidate)
    for file in ('native_job.json','gpu_job.json'):
        job=read(BASE/file);assert job['status']=='finished' and job['exit_code']==0
    assert native['passed'] and native['roots_and_metadata_released'] and len(native['cases'])==3
    assert {r['dtype'] for r in native['cases']}=={'torch.bfloat16','torch.float16','torch.float32'}
    for row in native['cases']:assert all(value is True for name,value in row.items() if name!='dtype')
    assert raw['passed'] and raw['hooks_canonical_method_restored'] and (raw['context'],raw['decode'])==(4096,12)
    a,b=raw['arms'];assert (a['arm'],b['arm'])==('cpu_control','cpu_gpu_indices')
    for field in ('predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes','host_initial_write_bytes'):assert a[field]==b[field],field
    assert len(a['frames'])==13 and len(a['selections'])==384 and len(a['predicted_token_ids'])==12
    assert a['canonical_reads_during_selector']==b['canonical_reads_during_selector']==0
    assert b['query_index_refreshes']==13 and b['query_index_reads']==b['selected_layers']==384
    assert b['query_index_metadata_h2d_bytes']==13*32*(1090+64)*(8*4+8)
    older=read(ROOT/'results/hardware/pool_query_selector_v1/gpu_v2.json');old=next(r for r in older['arms'] if r['arm']=='cpu_paired')
    for field in ('predicted_token_ids','frames','selections','host_read_bytes','host_write_bytes','host_initial_write_bytes'):assert a[field]==old[field],('frozen query replay',field)
    assert "args.cpu_batched_scores and args.strategy != 'lru'" in (ROOT/'experiments/evaluate_query_selection.py').read_text()
    output=BASE/'verified_correctness.json';assert not output.exists()
    result=dict(verified=True,tests_passed=283,focused_cpu_tests=10,native_dtypes=3,
        exact_native_query_prediction_steps=12,exact_cache_logit_counter_frames=13,exact_visible_kv_choice_observations=384,
        exact_frozen_native_query_replay=True,native_h2d_kv_bytes=b['host_read_bytes'],native_d2h_kv_bytes=b['host_write_bytes'],
        candidate_index_metadata_h2d_bytes=b['query_index_metadata_h2d_bytes'],index_refreshes=13,index_reads=384,
        stale_and_unconsumed_guards_before_dispatch=True,serial_streams_with_explicit_caller_handoff=True,
        pool_registration_switches_exact=True,pinned_roots_and_index_metadata_released=True,
        canonical_cpu_reads_during_selector=0,ordinary_source_cli_defaults_unchanged=True,
        ready_for_fresh_resource_gate=True,ordinary_adoption=False,full_goal_complete=False,
        limits='Single model/seed4K12 exact correctness. GPU stream cases require caller handoff for queued GPU writes. '
            'Index metadata increases H2D/memory and is charged separately; full coldKV scan is unchanged. No speed, '
            'independent quality, full512MiB/matchedfaithful-baseline or physicalSTT claim.',
        evidence_sha256={f:sha(BASE/f) for f in ('plan.json','control_source.tar.gz','implementation.json','implementation_source.tar.gz',
            'native.json','native_job.json','native.log','gpu.json','gpu_job.json','gpu.log','tests_focused.log','tests_focused_v2.log','tests_full.log','verify_correctness.py')})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence_sha256'},indent=2))
if __name__=='__main__':main()
