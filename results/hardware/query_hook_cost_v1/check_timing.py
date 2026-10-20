"""ThreeABBA no-op query math over already-visibleKV, with zero extra reads."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from transformers.models.mistral.modeling_mistral import rotate_half
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from pool_query_selector import PoolQuerySelector
from compare_cpu_batched import digest
from bulk_promotion_policy import BulkPromotionPolicy
BASE=Path(__file__).resolve().parent


class VisibleQueryMath(PoolQuerySelector):
    def _hook(self,index):
        def select(module,args,options):
            hidden=options.get('hidden_states',args[0] if args else None)
            if hidden is None or hidden.shape[1]!=1 or index not in self.policy.layer_caches:return
            cache=self.policy.layer_caches[index];quota=len(self.policy._vis_stt_pos.get(index,[]))
            if not quota:return
            keep=self.policy._vis_sram_keep[index];layer=options['past_key_values'].layers[index]
            query=module.q_proj(hidden).view(1,1,module.config.num_attention_heads,module.head_dim).transpose(1,2)
            cos,sin=options['position_embeddings'];query=query*cos.unsqueeze(1)+rotate_half(query)*sin.unsqueeze(1)
            query=query[0,:,0,:].float().view(cache.cfg.num_heads,module.num_key_value_groups,module.head_dim)
            cold_key=layer.keys[0,:,len(keep):,:];cold_value=layer.values[0,:,len(keep):,:]
            assert cold_key.shape[1]==quota
            fast_key=cache.sram_k[:,keep,:];fast_value=cache.sram_v[:,keep,:]
            weights=torch.softmax(torch.cat((torch.einsum('ghd,gnd->ghn',query,fast_key.float()),
                torch.einsum('ghd,gnd->ghn',query,cold_key.float())),dim=-1)*module.scaling,dim=-1)[...,fast_key.shape[1]:].mean((0,1))
            chosen=torch.topk(weights,quota).indices.tolist();chosen.sort()
            assert chosen==list(range(quota))
            relative=torch.tensor(chosen,device=cold_key.device,dtype=torch.long)
            layer.keys=torch.cat((fast_key,cold_key.index_select(1,relative)),dim=1).unsqueeze(0).to(layer.keys.dtype)
            layer.values=torch.cat((fast_value,cold_value.index_select(1,relative)),dim=1).unsqueeze(0).to(layer.values.dtype)
            self.selected_layers+=1
        return select


def main():
    ref=json.loads((ROOT/json.loads((BASE/'plan.json').read_text())['control']).read_text())
    path=ROOT/'results/hardware/pool_query_selector_v1/check_timing.py'
    spec=importlib.util.spec_from_file_location('scoped_query_timing',path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    # Reuse unchanged existing timing scopes, warmup, ABBA and quiet checks.
    driver.__file__=str(Path(__file__));driver.PoolQuerySelector=VisibleQueryMath
    original_profile=driver.profile_one
    def profile(model,tokenizer,policy,*args,**options):
        result=original_profile(model,tokenizer,policy,*args,**options)
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for field,reference in [('host_read_bytes',ref['prototype_counts']['host_read_bytes']),
                ('host_write_bytes',ref['prototype_counts']['host_write_bytes'])]:
            assert sum(getattr(c,field) for c in policy.layer_caches.values())==reference,field
        result.update(state_sha256=state,logical_counters=counters,ordinary_state_transfers_exact=True)
        return result
    driver.profile_one=profile
    driver.main()
if __name__=='__main__':main()
