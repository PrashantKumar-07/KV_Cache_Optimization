"""Reused development generation, current capacities, identical arms/leases."""
import hashlib,json,sys,types
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'experiments'))
from cpu_phase_policy import CpuPhasePolicy
from bulk_payload_policy import BulkPayloadPolicy
from shared_decode_transition import SharedAppendLease
B=Path(__file__).resolve().parent
plan=json.loads((B/'generation_plan.json').read_text());helper=R/'experiments/check_reuse_integrity.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==plan['helper_sha256']
source=helper.read_text();old='in ((991,1090),(1024,2048))';assert source.count(old)==1
source=source.replace(old,'in ((649,1982),)')
driver=types.ModuleType('scoped_bulk_generation');driver.__file__=str(helper);exec(compile(source,str(helper),'exec'),driver.__dict__)
class Control(driver.Trajectory,CpuPhasePolicy):
 def __init__(self,**options):
  self.trajectory,self.frames=hashlib.sha256(),0;super().__init__(**options)
class Candidate(driver.Trajectory,BulkPayloadPolicy):
 counts={}
 def __init__(self,**options):
  self.trajectory,self.frames=hashlib.sha256(),0;super().__init__(**options)
 def reset(self):
  type(self).counts=dict(bulk_steps=self.bulk_payload_steps,bulk_fallback_steps=self.bulk_payload_fallback_steps,cpu_append_calls=self.cpu_append_calls,cpu_fallback_steps=self.cpu_fallback_steps,cpu_reclaims=sum(c.cpu_reclaims for c in self.layer_caches.values()))
  super().reset()
def main():
 original=driver.measure;write_original=driver.atomic_json;leases=[]
 def measure(model,*args,**options):
  chosen=options['policy_cls'];own=[];before=len(model._forward_pre_hooks),len(model._forward_hooks)
  def factory(**kw):
   p=chosen(**kw);l=SharedAppendLease(model,p);own.append(l);leases.append(l);return p
  options['policy_cls']=factory
  try:
   row=original(model,*args,**options);row.update(append_fast_updates=sum(l.fast_updates for l in own),append_fallback_updates=sum(l.fallback_updates for l in own));return row
  finally:
   for l in own:l.close()
   assert before==(len(model._forward_pre_hooks),len(model._forward_hooks))
 def write(path,report):
  report.update(actual_control='cpu_phase',actual_candidate='bulk_payload_controller',owned_leases_closed=all(not l.handles and not l.pending for l in leases));write_original(path,report)
 driver.Control,driver.Candidate,driver.measure,driver.atomic_json=Control,Candidate,measure,write
 sys.argv=['generation','--manifest',str(B/'generation_manifest.json'),'--output',str(B/'generation.json')]
 try:driver.main()
 finally:
  for l in leases:l.close()
if __name__=='__main__':main()
