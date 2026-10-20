"""Independent expanded reference and pooling-order regression proof."""
import math
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from head_max_retention import head_max_importance,use_head_max
import longbench_eval as bench


@pytest.mark.parametrize('tokens,width,chunk',[(1,1,1),(13,5,2),(32,8,3),(41,13,128)])
@pytest.mark.parametrize('kernel',[1,5])
def test_matches_expanded_causal_pooled_head_max(tokens,width,chunk,kernel):
    torch.manual_seed(tokens)
    q=torch.randn(4,width,8);k=torch.randn(2,tokens,8)
    scores=q@k.repeat_interleave(2,0).transpose(-2,-1)/math.sqrt(8)
    future=torch.arange(tokens)[None,:]>torch.arange(tokens-width,tokens)[:,None]
    received=scores.masked_fill(future[None],float('-inf')).softmax(-1).sum(1)
    if kernel>1 and tokens>=kernel:
        received=torch.nn.functional.avg_pool1d(received[:,None],kernel,stride=1,padding=kernel//2)[:,0,:tokens]
    expected=received.amax(0)
    actual=head_max_importance(q,k,kernel,chunk)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    assert torch.equal(actual.argsort(descending=True),expected.argsort(descending=True))


def test_pool_each_head_before_maximum():
    # Two heads spike at different tokens. Pool(max(heads)) would combine them.
    q=torch.tensor([[[10.]], [[-10.]]]);k=torch.tensor([[[1.],[-1.],[0.],[0.],[0.]]])
    raw=(q@k.transpose(-2,-1)).softmax(-1).sum(1)
    expected=torch.nn.functional.avg_pool1d(raw[:,None],3,stride=1,padding=1)[:,0].amax(0)
    wrong=torch.nn.functional.avg_pool1d(raw.amax(0)[None,None],3,stride=1,padding=1).flatten()
    assert not torch.allclose(expected,wrong)
    torch.testing.assert_close(head_max_importance(q,k,3,1),expected)


def test_experimental_patch_restores_on_exception():
    original=bench.gqa_causal_importance
    with pytest.raises(RuntimeError):
        with use_head_max():
            assert bench.gqa_causal_importance is head_max_importance
            raise RuntimeError('synthetic runner failure')
    assert bench.gqa_causal_importance is original
