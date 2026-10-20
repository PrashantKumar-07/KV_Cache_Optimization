"""Offline stable8-page recall on unchanged native full-scan query trajectory."""
import hashlib,importlib.util,json,math,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.mistral.modeling_mistral import rotate_half
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from pool_query_selector import PoolQuerySelector
from provenance import atomic_json,runtime_identity
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_query_gpu_driver',ROOT/'results/hardware/pool_query_selector_v1/check_gpu.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class AuditedSelector(PoolQuerySelector):
    records=[]
    def __init__(self,model,policy):
        self.groups={};super().__init__(model,policy)
    def _hook(self,index):
        original=super()._hook(index)
        def select(module,args,options):
            result=original(module,args,options)
            hidden=options.get('hidden_states',args[0] if args else None)
            if hidden is None or hidden.shape[1]!=1 or index not in self.policy.layer_caches:return result
            cache=self.policy.layer_caches[index];live=torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)
            quota=len(self.policy._vis_stt_idx.get(index,[]))
            if not quota or len(live)<=quota:return result
            mapping,next_page,fill=self.groups.get(index,({},0,0))
            if not mapping:
                mapping={pos:row//16 for row,pos in enumerate(cache.stt_pos)}
                next_page=math.ceil(len(cache.stt_pos)/16);fill=0
            for pos in cache.stt_pos:
                if pos not in mapping:
                    mapping[pos]=next_page;fill+=1
                    if fill==16:next_page+=1;fill=0
            self.groups[index]=(mapping,next_page,fill)
            query=module.q_proj(hidden).view(1,1,module.config.num_attention_heads,module.head_dim).transpose(1,2)
            cos,sin=options['position_embeddings'];query=query*cos.unsqueeze(1)+rotate_half(query)*sin.unsqueeze(1)
            query=query[0,:,0,:].float().view(cache.cfg.num_heads,module.num_key_value_groups,module.head_dim)
            before=cache.host_read_bytes
            keys,_=cache.select_stt_pair(live.tolist())
            diagnostic_bytes=cache.host_read_bytes-before
            fast=cache.sram_k[:,self.policy._vis_sram_keep[index],:].float()
            weights=torch.softmax(torch.cat((torch.einsum('ghd,gnd->ghn',query,fast),
                torch.einsum('ghd,gnd->ghn',query,keys.float())),dim=-1)*module.scaling,dim=-1)[...,fast.shape[1]:].mean((0,1))
            oracle=torch.topk(weights,quota).indices
            chosen=sorted(live[oracle].tolist());assert chosen==self.policy._vis_stt_idx[index]
            page_ids=torch.tensor([mapping[cache.stt_pos[i]] for i in live.tolist()],device=keys.device)
            active,inverse=torch.unique(page_ids,sorted=True,return_inverse=True)
            sums=torch.zeros(keys.shape[0],len(active),keys.shape[2],device=keys.device,dtype=torch.float32)
            sums.index_add_(1,inverse,keys.float());counts=torch.bincount(inverse,minlength=len(active))
            centers=sums/counts[None,:,None]
            scores=torch.einsum('ghd,gpd->ghp',query,centers).amax((0,1))
            ranked=torch.argsort(scores,descending=True)
            n=min(8,len(active))
            while n<len(active) and int(counts[ranked[:n]].sum())<quota:n+=1
            rows=torch.where(torch.isin(inverse,ranked[:n]))[0]
            assert len(rows)>=quota
            selected=rows[torch.topk(weights[rows],quota).indices]
            full_mass=float(weights[oracle].sum());selected_mass=float(weights[selected].sum())
            overlap=int(torch.isin(selected,oracle).sum())
            type(self).records.append(dict(layer=index,step=len(type(self).records)//len(self.policy.layer_caches),
                live_rows=len(live),quota=quota,active_pages=len(active),pages_scanned=n,candidate_rows=len(rows),
                oracle_overlap=overlap,full_oracle_mass=full_mass,selected_full_distribution_mass=selected_mass,
                mass_ratio=selected_mass/full_mass,centroid_summary_bytes=sums.numel()*sums.element_size()+counts.numel()*counts.element_size(),
                diagnostic_native_h2d_bytes=diagnostic_bytes,underfilled=len(rows)<quota))
            return result
        return select


@torch.inference_mode()
def main():
    output=BASE/'recall.json';assert not output.exists()
    plan=json.loads((BASE/'plan.json').read_text());ref=json.loads((ROOT/plan['reference']).read_text())
    assert hashlib.sha256((ROOT/plan['reference']).read_bytes()).hexdigest()==plan['reference_sha256']
    report=dict(passed=False,records=[],runtime=runtime_identity(),plan_sha256=hashlib.sha256((BASE/'plan.json').read_bytes()).hexdigest())
    original=driver.PoolQuerySelector
    try:
        torch.manual_seed(0)
        model=AutoModelForCausalLM.from_pretrained('../models/mistral-7b-instruct',torch_dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='eager').eval()
        tokenizer=AutoTokenizer.from_pretrained('../models/mistral-7b-instruct')
        prompt=Path('results/remediation_v3/latency_prompt.txt').read_text()
        ids=tokenizer(prompt,truncation=False,return_tensors='pt').input_ids[:,:16384].to('cuda')
        assert ids.shape==(1,16384)
        input_sha=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest();assert input_sha==ref['input_ids_sha256']
        driver.PoolQuerySelector=AuditedSelector
        result=driver.run(model,ids,ref['teacher_token_ids'],'cpu_paired')
        assert result['predicted_token_ids']==ref['predicted_token_ids']
        assert result['frames'][-1]['state_sha256']==ref['state_sha256']
        assert result['frames'][-1]['counters']==ref['counters']
        assert result['canonical_reads_during_selector']==0
        assert len(AuditedSelector.records)==1600
        report.update(passed=True,input_ids_sha256=input_sha,result=result,records=AuditedSelector.records,
            own_hooks_canonical_method_restored=True,limits=plan['limits'])
    except BaseException as e:report.update(error=f'{type(e).__name__}: {e}',records=AuditedSelector.records);raise
    finally:driver.PoolQuerySelector=original;atomic_json(output,report)
    print('Full16K50 native query replay exact;1600offline stable8page recall rows recorded',flush=True)
if __name__=='__main__':main()
