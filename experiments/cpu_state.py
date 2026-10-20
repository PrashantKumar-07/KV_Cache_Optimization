"""Research fixed CPU owners and native whole demotion/reclaim phases."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sysconfig
import numpy as np
import torch

@lru_cache(maxsize=1)
def kernel():
    source=Path(__file__).with_name('native_cpu_state.cpp');digest=hashlib.sha256(source.read_bytes()).hexdigest()
    build=Path('/data/nishant/tieredkv_external')/('cpu_state_'+digest[:16]);build.mkdir(parents=True,exist_ok=True)
    binary=build/'tieredkv_cpu_state_v1.so';manifest=build/'manifest.json'
    identity=dict(source_sha256=digest,python_abi=sysconfig.get_config_var('SOABI'))
    if not binary.exists():
        temporary=build/'tieredkv_cpu_state_v1.pending.so'
        subprocess.run(['g++','-std=c++17','-shared','-fPIC','-O2','-fno-fast-math','-I'+str(Path(torch.__file__).parent/'include'),'-I'+sysconfig.get_path('include'),str(source),'-o',str(temporary)],check=True,capture_output=True,text=True)
        temporary.replace(binary)
        manifest.write_text(json.dumps(dict(**identity,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
    proof=json.loads(manifest.read_text())
    if any(proof.get(k)!=v for k,v in identity.items()) or proof['binary_sha256']!=hashlib.sha256(binary.read_bytes()).hexdigest():raise ValueError('CPU state build provenance mismatch')
    spec=importlib.util.spec_from_file_location('tieredkv_cpu_state_v1',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class CpuStateArena:
    def __init__(self,layers,hot_capacity,cold_capacity):
        if any(type(n) is not int or n<1 or n>np.iinfo(np.int32).max for n in [layers,hot_capacity,cold_capacity]):raise ValueError('Positive bounded CPU geometry required')
        self.owners={}
        for name,dtype,width in [('hot',np.int64,hot_capacity),('grace',np.bool_,hot_capacity),('cold',np.int64,cold_capacity),('shadow',np.bool_,cold_capacity),('resident',np.bool_,cold_capacity),('streak',np.float32,cold_capacity),('slots',np.int32,cold_capacity),('free',np.int32,cold_capacity)]:
            self.owners[name]=np.zeros((layers,width),dtype=dtype)
        for name,dtype in [('hot_n',np.int32),('cold_n',np.int32),('free_n',np.int32),('version',np.int64)]:self.owners[name]=np.zeros(layers,dtype=dtype)
        self.owners['free'][:]=np.arange(cold_capacity,dtype=np.int32)
        self.owners['free_n'][:]=cold_capacity
        self.epoch=0
        self._command_owner=object()
    def load(self,layer,hot,cold,grace,shadow,resident,streak,slots,free):
        """Fixture/initialization boundary only; supported native phases retain roots."""
        if type(layer) is not int or not 0<=layer<len(self.owners['hot_n']):raise ValueError('Invalid layer')
        if not 0<=self.owners['version'][layer]<np.iinfo(np.int64).max:raise ValueError('CPU version exhausted')
        values=dict(hot=hot,cold=cold,grace=grace,shadow=shadow,resident=resident,streak=streak,slots=slots,free=free)
        converted={}
        h,c=len(hot),len(cold)
        if len(grace)!=h or any(len(values[n])!=c for n in ['shadow','resident','streak','slots']) or c+len(free)!=self.owners['cold'].shape[1]:raise ValueError('Invalid logical geometry')
        for name,value in values.items():
            a=np.asarray(value)
            if a.ndim!=1 or len(a)>self.owners[name].shape[1]:raise ValueError('CPU fixture exceeds owner')
            if name in ['hot','cold','slots','free'] and len(a) and (not np.issubdtype(a.dtype,np.integer) or np.any(a<np.iinfo(self.owners[name].dtype).min) or np.any(a>np.iinfo(self.owners[name].dtype).max)):raise ValueError('Invalid integer state')
            converted[name]=np.asarray(a,dtype=self.owners[name].dtype)
        partition=np.concatenate((converted['slots'],converted['free']))
        if not np.array_equal(np.sort(partition),np.arange(self.owners['cold'].shape[1],dtype=np.int32)):raise ValueError('Invalid slot partition')
        for name,a in converted.items():self.owners[name][layer,:len(a)]=a
        for name,n in [('hot_n',h),('cold_n',c),('free_n',len(free))]:self.owners[name][layer]=n
        self.owners['version'][layer]+=1;self.epoch+=1
    def view(self,name,layer):
        count='hot_n' if name in ['hot','grace'] else 'free_n' if name=='free' else 'cold_n'
        return self.owners[name][layer,:int(self.owners[count][layer])]
    def demote(self,victims,counts,inclusive=True):
        if type(inclusive) is not bool:raise ValueError('Invalid demotion contract')
        result=kernel().demote(self.owners,victims,counts,inclusive);self.epoch+=1
        for a in result.values():a.flags.writeable=False
        result['_owner']=self._command_owner
        return result
    def promote(self,rows,counts,inclusive=True,floor=0):
        if type(inclusive) is not bool or type(floor) is not int or not 0<=floor<=np.iinfo(np.int32).max:raise ValueError('Invalid promotion contract')
        result=kernel().promote(self.owners,rows,counts,inclusive,floor);self.epoch+=1
        for a in result.values():a.flags.writeable=False
        result['_owner']=self._command_owner
        return result
    def validate_commands(self,commands):
        versions=commands.get('versions')
        if commands.get('_owner') is not self._command_owner or not isinstance(versions,np.ndarray) or versions.dtype!=np.int64 or versions.shape!=self.owners['version'].shape or not np.array_equal(versions,self.owners['version']):raise ValueError('Stale or foreign CPU state commands')
    def reclaim(self,keeps,counts):
        kernel().reclaim(self.owners,keeps,counts);self.epoch+=1
    def append_hot(self,positions,counts):
        kernel().append_hot(self.owners,positions,counts);self.epoch+=1
    @property
    def owned_bytes(self):return sum(a.nbytes for a in self.owners.values())
