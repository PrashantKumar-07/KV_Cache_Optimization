"""Fresh ordinary shared resource caller; CPU timestamp metadata charged."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import audit_bulk_resources as audit
from cpu_lru_metadata import CpuLruPolicy
BASE=Path(__file__).resolve().parent

def main():
    output=BASE/'resource.json';assert not output.exists()
    original=(audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv)
    final={}
    class ResourcePolicy(CpuLruPolicy):
        def __init__(self,**options):
            self.cpu_metadata_tensor_peak=0;self.cpu_metadata_boundaries=[]
            super().__init__(**options)
        def _visible_tuple(self,past):
            caches=list(self.layer_caches.values())
            timestamp=sum(c.stt_last_access.numel()*c.stt_last_access.element_size() for c in caches)
            packed=len(caches)*max((c.stt_last_access.numel() for c in caches),default=0)*4
            owned=timestamp+packed;self.cpu_metadata_tensor_peak=max(self.cpu_metadata_tensor_peak,owned)
            self.cpu_metadata_boundaries.append(dict(timestamp_tensor_bytes=timestamp,temporary_packed_tensor_bytes=packed,new_owned_cpu_metadata_tensor_bytes=owned))
            return super()._visible_tuple(past)
        def reset(self):
            final.update(timestamp_h2d_bytes=getattr(self,'timestamp_h2d_bytes',0),timestamp_refreshes=getattr(self,'timestamp_refreshes',0),timestamp_initial_d2h_bytes=sum(getattr(c,'timestamp_initial_d2h_bytes',0) for c in self.layer_caches.values()),cpu_metadata_tensor_peak_bytes=self.cpu_metadata_tensor_peak,cpu_metadata_boundary_rows=self.cpu_metadata_boundaries.copy(),mirrors_released=all(not hasattr(c,'_visible_access_gpu') for c in self.layer_caches.values()),timestamp_residency_cpu=all(c.stt_last_access.device.type=='cpu' for c in self.layer_caches.values()),prototype_counts={f:sum(getattr(c,f,0) for c in self.layer_caches.values()) for f in ['host_read_bytes','host_write_bytes','host_initial_write_bytes']})
            final['prototype_counts'].update(native_h2d_calls=getattr(self.manager,'h2d_calls',0),native_d2h_calls=getattr(self.manager,'d2h_calls',0))
            super().reset()
    def write(path,report):
        report.update(final,cpu_metadata_resource_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_metadata_limits='New CPUmetadata tensor-root byte bound includes timestampvectors and boundary packedbuffer. CPU malloc/allocatorrounding/RSS/otherhostretention unmeasured; not completeprocessmemory. NativeKVcounts exclude separatelyreported timestampcopies. Existing freshmeasurement boundaries unchanged.')
        original[1](path,report)
    def run():
        saved=sys.argv;sys.argv=[*saved,'--prefill-last-logits']
        try:original[2]()
        finally:sys.argv=saved
    audit.BulkPromotionPolicy=ResourcePolicy;audit.fresh.atomic_json=write;audit.fresh.main=run
    sys.argv=['audit','--kind','fresh','--arm','bulk','--hot','991','--cold','1090','--output',str(output)]
    try:audit.main()
    finally:audit.BulkPromotionPolicy,audit.fresh.atomic_json,audit.fresh.main,sys.argv=original
if __name__=='__main__':main()
