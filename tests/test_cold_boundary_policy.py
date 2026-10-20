import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import cold_boundary_policy as module
from bulk_payload_policy import BulkPayloadPolicy

def test_preflight_rejection_calls_original_once(monkeypatch):
 p=module.ColdBoundaryPolicy();past=object();calls=[]
 def reject(policy,raw):
  assert policy is p and raw is past
  raise ValueError('unsupported before mutation')
 monkeypatch.setattr(module,'visible',reject);monkeypatch.setattr(BulkPayloadPolicy,'_visible_tuple',lambda policy,raw:calls.append((policy,raw)) or 'original')
 assert p._visible_tuple(past)=='original' and calls==[(p,past)] and p.cold_boundary_fallback_calls==1

def test_native_success_does_not_enter_original(monkeypatch):
 p=module.ColdBoundaryPolicy();past=object();result=object()
 monkeypatch.setattr(module,'visible',lambda policy,raw:result)
 def wrong(*args):raise AssertionError('Unexpected original path')
 monkeypatch.setattr(BulkPayloadPolicy,'_visible_tuple',wrong)
 assert p._visible_tuple(past) is result and p.cold_boundary_fallback_calls==0
