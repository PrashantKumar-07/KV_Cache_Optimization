"""Executable root verification; independent review remains pending."""
from pathlib import Path
import datetime,hashlib,json,sys,tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from provenance import model_artifact_identity
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
target=BASE/'verified_parity.json';assert not target.exists()
p=json.loads((BASE/'plan.json').read_text());r=json.loads((BASE/'parity.json').read_text())
assert r['passed'] and r['plan_sha256']==sha(BASE/'plan.json')
assert sha(ROOT/p['runner'])==p['runner_sha256']
assert sha(ROOT/p['source_archive'])==p['source_archive_sha256']
with tarfile.open(ROOT/p['source_archive'],'r:gz') as tf:
 source={m.name:hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf if m.isfile()}
assert source==p['source_sha256']
assert all(sha(ROOT/n)==h for n,h in source.items())
assert model_artifact_identity(p['model'])==p['model_identity']
for name in ['prompt','teacher_stream']:assert sha(ROOT/p[name])==p[name+'_sha256']
trajectory=json.loads((ROOT/'results/audit_20261013/current_snap_resource_trajectory/diagnostic.json').read_text())
assert r['input_ids_sha256']==trajectory['input_ids_sha256']
a,b=r['arms']['reference'],r['arms']['reused']
assert a['full_vocab_arrays']==b['full_vocab_arrays']==51
assert a['KV_trace']==b['KV_trace'] and len(a['KV_trace'])==51
assert a['hooks_and_groups_restored'] and b['hooks_and_groups_restored']
assert a['projection_counts']==dict(q=1664,k=1664)
assert b['projection_counts']==dict(q=1632,k=1632)
for step,rows in enumerate(a['KV_trace']):
 assert len(rows)==32 and all(x['shape']==[1,32,904+step,128] for x in rows)
pred=[]
with np.load(BASE/'reference_logits.npz') as a,np.load(BASE/'reused_logits.npz') as b:
 assert set(a.files)==set(b.files)=={str(i) for i in range(51)}
 for i in range(51):
  x,y=a[str(i)],b[str(i)];assert x.shape==y.shape==(1,1,32000)
  assert np.isfinite(x).all() and np.isfinite(y).all() and x.tobytes()==y.tobytes()
  pred.append(int(x[:,-1].argmax(-1).item()))
assert pred[1:]==trajectory['arms']['snap']['predicted_token_ids']
out={'verified':True,'reviewer':'root-agent executable proof; independent review pending',
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files':len(source),
 'finite_full_vocabulary_arrays_byte_exact':51,'layers_per_boundary_KV_hashes_exact':32,
 'cache_boundaries_byte_hash_exact':51,'predictions_match_previous_faithful_snap_trajectory':True,
 'removed_projections_per_16K_prefill':{'Q':32,'K':32},'native_decode_projection_counts_unchanged':True,
 'hooks_and_groups_restored':True,'default_changed':False,'timing_valid':False,
 'scope':'One16Kprompt BF16 Mistral;51fixed teacherforwards lastlogits. No performance/quality/SOTA claim, no headnormalization/slidingwindow/othermodel proof.',
 'evidence_sha256':{str(q.relative_to(ROOT)):sha(q) for q in [BASE/'plan.json',BASE/'parity.json',BASE/'reference_logits.npz',BASE/'reused_logits.npz',Path(__file__),ROOT/p['runner'],ROOT/p['source_archive']]}}
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
