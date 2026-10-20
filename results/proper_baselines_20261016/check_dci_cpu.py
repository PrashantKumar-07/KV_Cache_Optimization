import torch,numpy as np,ctypes,json,hashlib
from pathlib import Path
import dci
p=Path(dci.__file__);lib=ctypes.CDLL(str(p))
fn=lib.bf16_to_f32_array;fn.argtypes=[ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_float),ctypes.c_int]
bits=np.arange(65536,dtype=np.uint16);values=np.empty(65536,dtype=np.float32)
fn(bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),len(bits))
assert np.array_equal(values.view(np.uint32),bits.astype(np.uint32)<<16)
conv=lib.f32_to_bf16_array;conv.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_uint16),ctypes.c_int]
rng=np.random.default_rng(0);x=rng.standard_normal(10003).astype(np.float32);out=np.empty(len(x),np.uint16)
conv(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),len(x))
expected=torch.from_numpy(x).bfloat16().view(torch.int16).numpy().view(np.uint16)
assert np.array_equal(out,expected)
assert all(hasattr(dci,k) for k in ["PyDCI","copy_to_buffer","diff_pages_by_head","reuse_copy_node","reuse_update_node"])
report={"verified":True,"scope":"ActualcompiledofficialMDCIBF16CPUutilities andrequiredAPI,notindex/model/GPUproof","BF16_to_FP32_exact_patterns":65536,"FP32_to_BF16_RNE_matches_Torch":10003,"module_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"module":str(p),"torch":torch.__version__}
Path("dci_cpu_precision_proof.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))
