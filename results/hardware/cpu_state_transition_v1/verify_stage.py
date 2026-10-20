"""Read-only native CPU phase, frozen corpus and GPU handoff proof."""
import hashlib,json,tarfile
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def main():
 out=BASE/'verified_stage.json';assert not out.exists()
 for kind,plan in [('stage','plan.json'),('gpu','gpu_plan.json')]:
  frozen=read(BASE/(kind+'_implementation.json'))
  assert sha(BASE/plan)==frozen['plan_sha256'] and sha(BASE/(kind+'_source.tar.gz'))==frozen['archive_sha256']
  with tarfile.open(BASE/(kind+'_source.tar.gz')) as t:
   archived={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
  assert archived==frozen['sources'] and all(sha(ROOT/f)==h for f,h in archived.items())
 assert read(BASE/'suite_v3_job.json')['exit_code']==0 and '489 passed' in (BASE/'suite_v3.log').read_text()
 assert '30 tests collected' in (BASE/'collection.log').read_text()
 assert read(BASE/'gpu_job.json')['status']=='finished' and read(BASE/'gpu_job.json')['exit_code']==0
 replay=read(BASE/'replay.json');job=read(BASE/'replay_job.json')
 assert job['status']=='finished' and job['artifact_sha256']==sha(BASE/'replay.json')
 assert json.loads((BASE/'replay.log').read_text())==replay
 assert replay['passed'] and replay['counts']==dict(records=1600,paid_rows=2560,restored_backups=23872,batches=50)
 assert replay['owned_cpu_bytes']==1116896 and replay['roots_preserved'] and replay['exact_metadata_and_slots'] and replay['gpu_metadata_commands_exact']
 for f,h in replay['source_sha256'].items():assert sha(ROOT/f)==h
 corpus=ROOT/'results/hardware/controller_transition_corpus_v1'
 assert sha(corpus/'transitions.jsonl.gz')==replay['corpus_sha256'] and sha(corpus/'verified_corpus.json')==replay['corpus_proof_sha256']
 build=read(BASE/'native_build.json');assert sha(Path(build['binary']))==build['binary_sha256'] and sha(ROOT/'experiments/native_cpu_state.cpp')==build['source_sha256']
 gpu=read(BASE/'gpu.json');assert gpu['passed'] and len(gpu['rows'])==5
 assert [r['case'] for r in gpu['rows']]==['shadow','lru','floor_cover','exclusive','group8']
 for row in gpu['rows']:assert row['roots_padding_payload_f32_exact'] and len(row['outcomes'])==row['layers']
 expected={'shadow':[4,0,2,2],'lru':[4,2,2,0],'floor_cover':[2,2,0,0],'exclusive':[4,4,0,0],'group8':[4,0,2,2]}
 for row in gpu['rows']:assert row['outcomes']==[expected[row['case']]]*row['layers']
 assert gpu['rows'][-1]['layers']==8 and gpu['rows'][-1]['scratch_peak_bytes']==229824
 for f,h in gpu['source_sha256'].items():assert sha(ROOT/f)==h
 primitive=read(ROOT/'results/hardware/pooled_state_transition_v1/verified_stage.json')
 assert sha(Path(primitive['binary']))==primitive['binary_sha256']
 result=dict(verified=True,tests_passed=489,focused_tests=30,recorded_transition_batches=50,recorded_layer_transitions=1600,paid_rows_replayed=2560,restored_backups_replayed=23872,cpu_declared_owner_bytes=1116896,cpu_declared_owner_mib=1116896/2**20,gpu_handoff_cases=5,gpu_handoff_layers=20,payload_f32_roots_padding_exact=True,atomic_malformed_input_and_stale_foreign_command_checks=True,ready_for_controller_integration=True,full_controller_integrated=False,ordinary_adoption=False,full_goal_complete=False,limits=replay['limits']+' '+gpu['limits']+' Promotion/hysteresis and hot/cold keep selection remain supplied decisions, not independently proven by the phase. GPU fixture uses the original cold selection. Full-model/native host-transfer, whole-loop timing and actual total memory gates remain required.',native_build=build,pooled_gpu_binary_sha256=primitive['binary_sha256'],evidence_sha256={f:sha(BASE/f) for f in ['plan.json','stage_implementation.json','stage_source.tar.gz','native_build.json','suite_v3_job.json','suite_v3.log','collection.log','replay_job.json','replay.json','replay.log','replay_corpus.py','gpu_plan.json','gpu_implementation.json','gpu_source.tar.gz','gpu_job.json','gpu.json','gpu.log','check_gpu_commands.py','verify_stage.py']})
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','native_build']},indent=2))
if __name__=='__main__':main()
