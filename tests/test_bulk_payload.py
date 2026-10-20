import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from bulk_payload import execute

def test_incomplete_batch_rejects_before_tensor_access():
    with pytest.raises(ValueError,match='Invalid payload batch'):execute([object()],[],[],[])
    with pytest.raises(ValueError,match='Invalid payload batch'):execute([],[],[],[])

def test_cpu_payload_cannot_reach_native_pointer_executor():
    state=SimpleNamespace(old_k=torch.zeros(2,3,4,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='CUDA payload geometry'):execute([state],[[0]],[[]],[[]])
