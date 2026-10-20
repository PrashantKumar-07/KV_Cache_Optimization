"""Experimental pooled-head maximum prompt ranking, same grouped KV layout."""
from contextlib import contextmanager
import math
import torch
import longbench_eval as bench


def head_max_importance(q_window,k_all,pool_kernel=5,query_chunk=16):
    q_heads,width,dim=q_window.shape
    k_heads,tokens,k_dim=k_all.shape
    if dim!=k_dim or q_heads%k_heads or width>tokens or query_chunk<1:
        raise ValueError('Invalid GQA importance dimensions or chunk size')
    groups=q_heads//k_heads
    queries=q_window.reshape(k_heads,groups,width,dim)
    key_t=k_all[:,None].transpose(-2,-1)
    received=torch.zeros(k_heads,groups,tokens,device=q_window.device,dtype=q_window.dtype)
    keys=torch.arange(tokens,device=q_window.device)
    for start in range(0,width,query_chunk):
        end=min(start+query_chunk,width)
        scores=torch.matmul(queries[:,:,start:end],key_t)/math.sqrt(dim)
        positions=torch.arange(tokens-width+start,tokens-width+end,device=q_window.device)
        scores.masked_fill_(keys[None,None,None,:]>positions[None,None,:,None],float('-inf'))
        received+=torch.softmax(scores,dim=-1).sum(dim=2)
    per_head=received.reshape(q_heads,tokens)
    if pool_kernel>1 and tokens>=pool_kernel:
        per_head=torch.nn.functional.avg_pool1d(per_head[:,None,:],kernel_size=pool_kernel,
            stride=1,padding=pool_kernel//2)[:,0,:tokens]
    return per_head.amax(dim=0)


@contextmanager
def use_head_max():
    """Serial experimental runner only; restore the original selector on exit."""
    original=bench.gqa_causal_importance
    bench.gqa_causal_importance=head_max_importance
    try:
        yield
    finally:
        bench.gqa_causal_importance=original
