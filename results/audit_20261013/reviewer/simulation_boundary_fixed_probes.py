"""Independent post-fix CPU proof; preserves before-fix probes/output."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
import torch
from tiered_kv_cache import TieredKVCache,TieredConfig
from cost_model import CostModel

torch.set_num_threads(1)
torch.manual_seed(71)
rows=[]
for hot,cold,n in [(torch.float32,torch.float8_e4m3fn,36),(torch.float32,torch.bfloat16,36),
                   (torch.bfloat16,torch.float8_e4m3fn,36),(torch.float32,torch.float32,36),
                   (torch.float32,torch.float8_e4m3fn,4),(torch.float32,torch.bfloat16,4)]:
    cfg=TieredConfig(num_heads=2,head_dim=4,sink_size=1,window_size=1,sram_capacity=4,
                     sttram_capacity=32,page_size=16,kv_dtype=hot,stt_kv_dtype=cold)
    c=TieredKVCache(cfg)
    x=torch.linspace(-1,1,2*n*4).reshape(2,n,4).to(hot)
    c.initial_bifurcation(x,x,x)
    initial_cold=len(c.stt_pos)
    pages=c._page_bounds()
    actual_bytes=sum(p[2].numel()*p[2].element_size()+p[3].numel()*p[3].element_size() for p in pages)
    calls=[]
    original=c.cm.sketch_read_cost
    def trace(*a,**kw):
        result=original(*a,**kw)
        calls.append({'elems':a[0],'kwargs':kw,'latency_us':result[0],'energy_nj':result[1]})
        return result
    c.cm.sketch_read_cost=trace
    q=torch.zeros(2,1,4,dtype=hot)
    c.step(q,q,q,n)
    lat=actual_bytes/1e9/c.cm.tiers['STT-RAM'].read_bw_gbps*1e6
    energy=actual_bytes*8*c.cm.tiers['STT-RAM'].read_pj_per_bit/1e3
    assert len(calls)==1
    assert math.isclose(calls[0]['latency_us'],lat,abs_tol=1e-15)
    assert math.isclose(calls[0]['energy_nj'],energy,abs_tol=1e-15)
    assert math.isclose(c.metrics.steps[-1].lat_sketch_us,lat,abs_tol=1e-15)
    rows.append({'hot':str(hot),'cold':str(cold),'initial_cold_rows':initial_cold,
                 'actual_bound_bytes':actual_bytes,'billed':calls[0]})

# Legacy omitted-width calls stay exactly identical to existing primitives.
compat=[]
for default in [1,2,4,8]:
    cm=CostModel(bytes_per_elem=default,tier_bytes_per_elem={'STT-RAM':1})
    for tier in ['STT-RAM','GPU_VRAM']:
        for elems in [0,1,37,1_000_000]:
            assert cm.sketch_read_cost(elems,tier)==(cm.read_latency_us(tier,elems),cm.read_energy_nj(tier,elems))
            for width in [1,2,4,8]:
                l,e=cm.sketch_read_cost(elems,tier,metadata_bytes_per_elem=width)
                assert math.isclose(l,elems*width/1e9/cm.tiers[tier].read_bw_gbps*1e6,abs_tol=1e-15)
                assert math.isclose(e,elems*width*8*cm.tiers[tier].read_pj_per_bit/1e3,abs_tol=1e-15)
    compat.append(default)
for bad in [0,-1,3,16]:
    try: cm.sketch_read_cost(1,metadata_bytes_per_elem=bad)
    except ValueError: pass
    else: raise AssertionError(bad)
cm=CostModel(bytes_per_elem=2)
assert cm.leakage_energy_nj('STT-RAM',524_288,1)==1_000_000.0
out={'cpu_only':True,'torch':torch.__version__,'seed':71,'threads':1,'all_checks_passed':True,
     'six_dtype_zero_cold_cases':rows,'legacy_compatibility_default_widths':compat,
     'one_MiB_one_second_STT_leakage_nj':cm.leakage_energy_nj('STT-RAM',524_288,1),
     'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
       ['src/cost_model.py','src/tiered_kv_cache.py','experiments/evaluate_query_selection.py','experiments/compare_query_latency.py']}}
print(json.dumps(out,indent=2))
