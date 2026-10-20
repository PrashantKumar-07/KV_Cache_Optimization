"""Research native CPU controller; existing Torch headers, no installation."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import subprocess
import sysconfig
from pathlib import Path
import torch

@lru_cache(maxsize=1)
def kernel():
    source=Path(__file__).with_suffix('.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()
    build=Path('/data/nishant/tieredkv_external')/('controller_'+digest[:16]);build.mkdir(parents=True,exist_ok=True)
    binary=build/'tieredkv_controller_v1.so';manifest=build/'manifest.json'
    identity=dict(source_sha256=digest,python_abi=sysconfig.get_config_var('SOABI'))
    if not binary.exists():
        temporary=build/'tieredkv_controller_v1.pending.so'
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2','-fno-fast-math',
            '-I'+str(Path(torch.__file__).parent/'include'),'-I'+sysconfig.get_path('include'),
            str(source),'-o',str(temporary)],check=True,capture_output=True,text=True)
        temporary.replace(binary)
        manifest.write_text(json.dumps(dict(**identity,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
    proof=json.loads(manifest.read_text())
    if any(proof.get(k)!=v for k,v in identity.items()) or proof['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():
        raise ValueError('Native controller source/ABI/binary differs from recorded build')
    spec=importlib.util.spec_from_file_location('tieredkv_controller_v1',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
