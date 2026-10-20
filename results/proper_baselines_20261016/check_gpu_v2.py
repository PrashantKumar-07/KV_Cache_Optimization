"""Official BF16 kernel and tiny non-offloaded model preflight, not a benchmark."""
from pathlib import Path
import hashlib,json,math,copy
import torch
from icecache import kernels,adapter
from icecache.infer_state import InferState
from transformers import MistralConfig,MistralForCausalLM
BASE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,atol=.04,rtol=.01):
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 difference=(a.float()-b.float());relative=(difference.norm()/b.float().norm().clamp(min=1e-12)).item()
 torch.testing.assert_close(a.float(),b.float(),atol=atol,rtol=rtol)
 assert relative<=.01
 return {'max_abs_error':difference.abs().max().item(),'relative_L2':relative}
def main():
 plan=json.loads((BASE/'gpu_preflight_v2_plan.json').read_text());output=BASE/'gpu_preflight_v2.json';assert not output.exists()
 for p,h in plan['sources_sha256'].items():assert sha(BASE/p)==h,p
 report={'execution_complete':False,'plan_sha256':sha(BASE/'gpu_preflight_v2_plan.json'),'kernels':[],'model_cases':[]}
 try:
  torch.manual_seed(0)
  with torch.inference_mode():
   for rows in (1,17,129):
    x=torch.randn(rows,128,device='cuda',dtype=torch.bfloat16);w=torch.randn(128,device='cuda',dtype=torch.bfloat16)
    expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).bfloat16()
    actual=kernels.rms_norm(x,w,1e-6);report['kernels'].append({'name':'rms_norm','rows':rows,**close(actual,expected)})
   for offset in (0,17,30000):
    q=torch.randn(1,17,32,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(1,17,8,128,device='cuda',dtype=torch.bfloat16)
    positions=torch.arange(offset,offset+17,device='cuda').float();freq=torch.arange(64,device='cuda').float()/64;angle=positions[:,None]/(10000.**freq)[None,:]
    cos=torch.cat((angle.cos(),angle.cos()),-1)[None,:,None,:];sin=torch.cat((angle.sin(),angle.sin()),-1)[None,:,None,:]
    def rotate(t):return torch.cat((-t[...,64:],t[...,:64]),-1)
    eq=(q.float()*cos+rotate(q.float())*sin).bfloat16();ek=(k.float()*cos+rotate(k.float())*sin).bfloat16()
    kernels.qk_apply_rotary_in_place(q,k,offset,1.,10000.)
    report['kernels'].append({'name':'rotary','offset':offset,'q':close(q,eq),'k':close(k,ek)})
   for page_size in (16,32):
    length=page_size+1;n_pages=2
    pool=torch.zeros(n_pages,2,page_size,8,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,length,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    indices=torch.tensor([0,1],device='cuda',dtype=torch.int32);indptr=torch.tensor([0,2],device='cuda',dtype=torch.int32);last=torch.tensor([1],device='cuda',dtype=torch.int32)
    kernels.append_paged_kv_cache(k,v,pool,indices,indptr,last)
    assert torch.equal(pool[:,0].reshape(-1,8,128)[:length],k[0]);assert torch.equal(pool[:,1].reshape(-1,8,128)[:length],v[0])
    nk=torch.randn(1,1,8,128,device='cuda',dtype=torch.bfloat16);nv=torch.randn_like(nk);last.fill_(2)
    kernels.append_paged_kv_cache(nk,nv,pool,indices,indptr,last);allk=torch.cat((k,nk),1);allv=torch.cat((v,nv),1)
    assert torch.equal(pool[:,0].reshape(-1,8,128)[:length+1],allk[0]);assert torch.equal(pool[:,1].reshape(-1,8,128)[:length+1],allv[0])
    work=torch.empty(16*1024*1024,device='cuda',dtype=torch.uint8);decoder=kernels.BatchDecodeWithPagedKVCacheWrapper(work,'NHD')
    decoder.begin_forward(indptr,last,32,8,128,page_size,data_type=torch.bfloat16)
    q=torch.randn(1,1,32,128,device='cuda',dtype=torch.bfloat16)
    try:actual=decoder.forward(q,pool,indices).reshape(1,1,32,128)
    finally:decoder.end_forward()
    eq=allk.transpose(1,2).repeat_interleave(4,1).float();ev=allv.transpose(1,2).repeat_interleave(4,1).float()
    expected=((q.transpose(1,2).float()@eq.transpose(-1,-2)/math.sqrt(128)).softmax(-1)@ev).transpose(1,2).bfloat16()
    report['kernels'].append({'name':'append+decode','page_size':page_size,'exact_prefill_and_append_payload':True,**close(actual,expected)})
   # All prompt/decode positions stay below the page budget: this checks the
   # model bridge without conflating intended sparse approximation with bugs.
   for length in (65,129):
    config=MistralConfig(vocab_size=128,hidden_size=1024,intermediate_size=2048,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=2,head_dim=128,sliding_window=None,max_position_embeddings=2048)
    reference=MistralForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval();patched=copy.deepcopy(reference)
    state=InferState(2,8,2,128,16,torch.bfloat16,torch.device('cuda'),page_budgets=16,n_max_pages=128,n_max_cpu_pages=128,n_sink_pages=2,n_win_pages=2)
    adapter.enable_icecache(patched,dtype=torch.bfloat16,device=torch.device('cuda'),infer_state=state)
    ids=torch.randint(0,128,(1,length),device='cuda');past=None;case=[]
    for step in range(6):
     tokens=ids if step==0 else torch.tensor([[step]],device='cuda');pos=torch.arange(length,device='cuda')[None] if step==0 else torch.tensor([[length+step-1]],device='cuda')
     a=reference(tokens,past_key_values=past,use_cache=True,position_ids=pos,logits_to_keep=1);past=a.past_key_values
     z=patched(tokens,use_cache=True,position_ids=pos,logits_to_keep=1)
     check=close(z.logits,a.logits,atol=.08,rtol=.02);check.update(step=step,argmax_equal=bool(torch.equal(z.logits.argmax(-1),a.logits.argmax(-1))));case.append(check);del a,z
    report['model_cases'].append({'prompt_tokens':length,'full_attention_only':True,'steps':case,'limitation':'No DCI offload or pretrained7B check yet; BF16 relativeL2<=1% and declared allclose gate, no bitwise equality claim.'})
    del reference,patched,state,past;torch.cuda.empty_cache()
   report.update(execution_complete=True,verified=True,scope='Compiled upstream BF16 kernels and tiny no-offload Mistral bridge only. Full7B/DCI-offload/fair quality/resources remain required; no performance/SOTA claim.')
 except BaseException as error:report['error']=f'{type(error).__name__}: {error}';raise
 finally:output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
