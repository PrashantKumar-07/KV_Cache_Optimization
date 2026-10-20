"""Research current-query cold selection through paired cache storage reads.

Full cold scan, serial batch-one decode. No ordinary CLI integration. Native
CPU pools transfer both K/V; account for those bytes before a latency claim.
"""
import torch
from transformers.models.mistral.modeling_mistral import rotate_half


class PoolQuerySelector:
    def __init__(self,model,policy):
        self.policy=policy
        self.handles=[]
        self.selected_layers=0
        self.scanned_rows=0
        try:
            for index,layer in enumerate(model.model.layers):
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    self._hook(index),with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()

    def _hook(self,index):
        def select(module,args,options):
            hidden=options.get('hidden_states',args[0] if args else None)
            if hidden is None or hidden.shape[1]!=1 or index not in self.policy.layer_caches:return
            cache=self.policy.layer_caches[index]
            old=self.policy._vis_stt_pos.get(index,[])
            if not old:return
            live=torch.where(~cache.stt_shadow)[0].to(cache.cfg.device)
            if len(live)<=len(old):return
            cos,sin=options['position_embeddings']
            query=module.q_proj(hidden).view(1,1,module.config.num_attention_heads,
                module.head_dim).transpose(1,2)
            query=query*cos.unsqueeze(1)+rotate_half(query)*sin.unsqueeze(1)
            query=query[0,:,0,:].float().view(cache.cfg.num_heads,
                module.num_key_value_groups,module.head_dim)
            keep=self.policy._vis_sram_keep[index]
            fast=cache.sram_k[:,keep,:].float()
            # Both tensors arrive on the model device through the storage API.
            cold_key,cold_value=cache.select_stt_pair(live.tolist())
            if cold_key.device!=query.device or cold_value.device!=query.device:
                raise ValueError('Paired cold reads must return tensors on the query device')
            fast_logits=torch.einsum('ghd,gnd->ghn',query,fast)
            slow_logits=torch.einsum('ghd,gnd->ghn',query,cold_key.float())
            weights=torch.softmax(torch.cat((fast_logits,slow_logits),dim=-1)*module.scaling,
                dim=-1)[...,fast.shape[1]:]
            ranking=weights.mean(dim=(0,1))
            chosen=live[torch.topk(ranking,len(old)).indices].tolist();chosen.sort()
            relative=torch.searchsorted(live,torch.tensor(chosen,device=live.device,dtype=live.dtype))
            layer=options['past_key_values'].layers[index]
            keys=torch.cat((cache.sram_k[:,keep,:],cold_key.index_select(1,relative)),dim=1)
            values=torch.cat((cache.sram_v[:,keep,:],cold_value.index_select(1,relative)),dim=1)
            if keys.shape[-2]!=layer.keys.shape[-2]:
                raise RuntimeError('Query selection changed shared causal-mask length')
            layer.keys=keys.unsqueeze(0).to(layer.keys.dtype)
            layer.values=values.unsqueeze(0).to(layer.values.dtype)
            self.policy._vis_stt_pos[index]=[cache.stt_pos[i] for i in chosen]
            self.policy._vis_stt_idx[index]=chosen
            self.selected_layers+=1
            self.scanned_rows+=len(live)
        return select
