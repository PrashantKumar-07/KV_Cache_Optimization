"""CPU-only read-only source probes; no device timing or accuracy conclusions."""
import hashlib, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from tiered_kv_cache import TieredConfig, TieredKVCache
from cost_model import CostModel

torch.set_num_threads(1)
torch.manual_seed(7)

def probe(cold_dtype):
    cfg = TieredConfig(num_heads=2, head_dim=4, sink_size=1, window_size=1,
                       sram_capacity=4, sttram_capacity=32, page_size=16,
                       stt_live_floor=0, kv_dtype=torch.bfloat16,
                       stt_kv_dtype=cold_dtype, device='cpu')
    cache = TieredKVCache(cfg)
    k = torch.randn(2,36,4).to(torch.bfloat16)
    cache.initial_bifurcation(k, k, k)
    accesses=[]
    orig=cache.sketch_key_tensor
    def trace():
        keys=orig()
        accesses.append({'elements':keys.numel(), 'bytes':keys.numel()*keys.element_size()})
        return keys
    cache.sketch_key_tensor=trace
    q=torch.ones(2,1,4,dtype=torch.bfloat16)
    first=cache._page_bounds()
    second=cache._page_bounds()
    assert all(torch.equal(a[2],b[2]) and torch.equal(a[3],b[3]) for a,b in zip(first,second))
    # mutate a live key directly: unchanged shape/positions, bounds must recompute
    row=int(torch.where(~cache.stt_shadow)[0][0])
    raw=cache.stt_k.float()
    raw[0,row,0]=100
    cache.stt_k=raw.to(cold_dtype)
    third=cache._page_bounds()
    assert not torch.equal(second[row//16][3], third[row//16][3])
    bound_elems=sum(a[2].numel()+a[3].numel() for a in third)
    bound_bytes=sum(a[2].numel()*a[2].element_size()+a[3].numel()*a[3].element_size() for a in third)
    billed_bytes=cache.cm.storage_bytes('STT-RAM',bound_elems)
    # actual step retains full-key reconstruction, bills two vectors per physical page
    accesses.clear()
    before=len(cache.stt_pos)
    cache.step(q, torch.zeros_like(q), torch.zeros_like(q), 36)
    rec=cache.metrics.steps[-1]
    expected=cache.cm.read_latency_us('STT-RAM',2*math.ceil(before/16)*2*4)
    assert math.isclose(rec.lat_sketch_us,expected)
    return {'cold_dtype':str(cold_dtype), 'initial_cold_rows':before,
            'identical_input_calls_each_reconstructed':3,
            'in_place_key_mutation_changes_bound':True,
            'step_full_key_tensor_accesses':accesses,
            'bound_elements':bound_elems,'computed_bound_bytes':bound_bytes,
            'analytical_bound_bytes':billed_bytes,'computed_bound_dtype':str(third[0][2].dtype),
            'full_K_bytes_to_billed_metadata_ratio':accesses[0]['bytes']/billed_bytes,
            'lat_sketch_us':rec.lat_sketch_us,
            'sketch_MACs':rec.sketch_macs}

# GB/s means decimal gigabytes; primitive units checked against exact examples.
cm=CostModel()
assert math.isclose(cm.read_latency_us('STT-RAM',500_000),20.0)
assert math.isclose(cm.read_energy_nj('STT-RAM',500_000),16_000.0)
assert math.isclose(cm.write_latency_us('STT-RAM',500_000),80.0)
assert math.isclose(cm.write_energy_nj('STT-RAM',500_000),64_000.0)
scenarios=[]
for label in cm.STT_BW_TENTPOLES:
    c=CostModel.with_stt_bw(label)
    scenarios.append({'label':label,'1MB_read_us':c.read_latency_us('STT-RAM',500_000),
                      '1MB_write_us':c.write_latency_us('STT-RAM',500_000),
                      '1MB_read_nj':c.read_energy_nj('STT-RAM',500_000),
                      '1MB_write_nj':c.write_energy_nj('STT-RAM',500_000)})
assert len({x['1MB_read_nj'] for x in scenarios})==1
assert len({x['1MB_write_nj'] for x in scenarios})==1
out={'CPU_only':True,'seed':7,'threads':1,'torch':torch.__version__,
     'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                     ['src/tiered_kv_cache.py','src/cost_model.py','src/metrics.py','src/baselines.py']},
     'probes':[probe(torch.bfloat16),probe(torch.float8_e4m3fn)],
     'dimensional_checks_passed':True,'bandwidth_scenarios':scenarios,
     'interpretation':'Tensor-access tracing demonstrates reconstruction semantics only; not physical traffic measurement. Full-K/bound ratio compares two alternative analytical boundaries; do not add full-K traffic to a declared ideal persistent-sketch model silently.'}
print(json.dumps(out,indent=2))
