"""Reuse the frozen lifetime observer for the native raw-buffer candidate."""
from pathlib import Path
import sys
import hashlib
import json
import audit_decode_kv_lifetimes as lifetime
from audit_snap_decode_lifetimes import CompactTrace
import native_reuse_visible as native
from provenance import atomic_json


class ObservedNative(lifetime.ObservedPolicy,native.NativeReusePolicy):
    """Existing observer delegates through the native reuse policy MRO."""


class NativeTraceKernel:
    def __init__(self,module):self.module=module

    def compact(self,*args):
        trace=ObservedNative.active
        with trace.phase('native_reuse_launch'):
            trace.sample('native_launch')
            return self.module.compact(*args)


def main():
    module=native.kernel()
    previous=(lifetime.ObservedPolicy,lifetime.StorageTrace,native.kernel)
    lifetime.ObservedPolicy=ObservedNative;lifetime.StorageTrace=CompactTrace
    native.kernel=lambda:NativeTraceKernel(module)
    try:
        lifetime.main()
        # The shared observer records the old fallback binary. Identify the
        # native writer separately, along with this scoped composition driver.
        output=Path(sys.argv[sys.argv.index('--output')+1])
        report=json.loads(output.read_text())
        report.update(native_reuse_composition=True,
            reuse_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            composition_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_json(output,report)
    finally:
        lifetime.ObservedPolicy,lifetime.StorageTrace,native.kernel=previous


if __name__=='__main__':main()
