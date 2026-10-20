"""Existing recorded fullmodel pair: ordinary bulk versus CPU timestamps."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import torch
import compare_reuse_visible as driver
from compare_bulk_promotion import RecordedBulkPolicy
from cpu_lru_metadata import CpuLruPolicy,visible_indices,EVICTION_AST
from longbench_eval import TieredKVPolicy
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent
class RecordedCpuLruPolicy(driver.RecordedNativePolicy,CpuLruPolicy):
    timestamp_counts={}
    def reset(self):
        type(self).timestamp_counts=dict(h2d_bytes=getattr(self,'timestamp_h2d_bytes',0),refreshes=getattr(self,'timestamp_refreshes',0),initial_d2h_bytes=sum(getattr(c,'timestamp_initial_d2h_bytes',0) for c in self.layer_caches.values()),cpu_timestamp_bytes=sum(c.stt_last_access.numel()*c.stt_last_access.element_size() for c in self.layer_caches.values()),mirrors_released=all(not hasattr(c,'_visible_access_gpu') for c in self.layer_caches.values()),all_timestamps_cpu=all(c.stt_last_access.device.type=='cpu' for c in self.layer_caches.values()))
        super().reset()
def main():
    native=driver.RecordedNativePolicy;control=driver.trial.FusedBoundedPolicy;write=driver.atomic_json;argv=sys.argv
    checks=[]
    for boolean in (False,True):
        for size in (1,8,32,1090):
            gpu=(torch.arange(size,device='cuda')%4).float()
            shadow=torch.zeros(size,dtype=torch.bool);shadow[::7]=True
            cache=SimpleNamespace(stt_last_access=gpu,stt_shadow=shadow)
            live=torch.where(~shadow)[0].tolist()
            expected=TieredKVPolicy._visible_cold_indices(cache,live,min(32,len(live)),boolean)
            cache.stt_last_access=gpu.cpu();cache._visible_access_gpu=cache.stt_last_access.cuda()
            actual=visible_indices(cache,live,min(32,len(live)),boolean)
            assert expected==actual
            checks.append(dict(boolean=boolean,size=size,gpu_tie_order_exact=True))
    def save(path,report):
        report.update(actual_control='ordinary_bulk',actual_candidate='cpu_lru_metadata',gpu_sort_tie_cases=checks,eviction_ast=EVICTION_AST,timestamp_counts=dict(RecordedCpuLruPolicy.timestamp_counts),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='Timestamp D2H/H2D are separate metadata payload, not KVtraffic. Frozen original recorded4K12 model pair; no independentquality/speed/resource/full512MiB/physicalSTT claim.')
        write(path,report)
    driver.RecordedNativePolicy=RecordedCpuLruPolicy;driver.trial.FusedBoundedPolicy=RecordedBulkPolicy;driver.atomic_json=save
    sys.argv=['check','--output',str(BASE/'gpu.json'),'--contexts','4096','--decode','12']
    try:driver.main()
    finally:driver.RecordedNativePolicy=native;driver.trial.FusedBoundedPolicy=control;driver.atomic_json=write;sys.argv=argv
if __name__=='__main__':main()
