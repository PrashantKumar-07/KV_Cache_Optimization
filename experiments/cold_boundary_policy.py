"""Default-off native cold boundary; preserve unsupported original paths."""
from bulk_payload_policy import BulkPayloadPolicy
from cold_boundary import visible

FIELDS=['cold_boundary_steps','cold_boundary_descriptor_h2d_bytes','cold_boundary_rank_peak_bytes','cold_boundary_cpu_peak_bytes','cold_boundary_fallback_calls']
class ColdBoundaryPolicy(BulkPayloadPolicy):
 def __init__(self,**options):
  super().__init__(**options)
  for name in FIELDS:setattr(self,name,0)
 def _visible_tuple(self,past):
  if any(c.pending_keep is not None for c in self.layer_caches.values()):
   self._compact_pending_hot()
  try:return visible(self,past)
  except ValueError:
   self.cold_boundary_fallback_calls+=1
   return super()._visible_tuple(past)
 def reset(self):
  super().reset()
  for name in FIELDS:setattr(self,name,0)
