"""Pinned CPU-DRAM <-> GPU transfer screen; this does not measure STT-MRAM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import atomic_json, runtime_identity


def link(device_index):
    fields='pcie.link.gen.current,pcie.link.width.current'
    try:
        return subprocess.check_output(
            ['nvidia-smi',f'--id={device_index}',f'--query-gpu={fields}',
             '--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        return 'unknown'


def trial(direction,host,gpu,parts,device):
    torch.cuda.synchronize(device)
    start=time.perf_counter()
    for begin,end in parts:
        if direction=='host_to_gpu':
            gpu[begin:end].copy_(host[begin:end],non_blocking=True)
        else:
            host[begin:end].copy_(gpu[begin:end],non_blocking=True)
    torch.cuda.synchronize(device)
    return 1000*(time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--reps',type=int,default=30)
    ap.add_argument('--output',default='results/hardware/host_transfer_probe.json')
    args=ap.parse_args()
    device=torch.device(args.device)
    if device.type!='cuda' or args.reps<2:
        ap.error('CUDA device and at least two repeats required')
    index=device.index if device.index is not None else torch.cuda.current_device()
    before=link(index)
    cases=[('one_layer_token',4096,1),
           ('32_layer_tokens_separate',4096*32,32),
           ('32_layer_tokens_packed',4096*32,1),
           ('one_layer_32_tokens',4096*32,1),
           ('32_layers_32_tokens_rows',4096*32*32,1024),
           ('32_layers_32_tokens_separate',4096*32*32,32),
           ('32_layers_32_tokens_packed',4096*32*32,1),
           ('full_cold_tier',256*1024*1024,1)]
    rows=[]
    for label,total,nparts in cases:
        host=torch.empty(total,dtype=torch.uint8,pin_memory=True)
        gpu=torch.empty(total,dtype=torch.uint8,device=device)
        host.fill_(0x5a)
        gpu.copy_(host,non_blocking=True)
        torch.cuda.synchronize(device)
        step=total//nparts
        parts=[(i*step,(i+1)*step) for i in range(nparts)]
        # Test both directions with distinct patterns before taking timings.
        gpu.zero_()
        trial('host_to_gpu',host,gpu,parts,device)
        if gpu[0].item()!=0x5a or gpu[-1].item()!=0x5a:
            raise AssertionError('host-to-GPU transfer mismatch')
        gpu.fill_(0xa5)
        trial('gpu_to_host',host,gpu,parts,device)
        if host[0].item()!=0xa5 or host[-1].item()!=0xa5:
            raise AssertionError('GPU-to-host transfer mismatch')
        host.fill_(0x5a)
        gpu.fill_(0x5a)
        for direction in ('host_to_gpu','gpu_to_host'):
            for _ in range(5):trial(direction,host,gpu,parts,device)
            times=[trial(direction,host,gpu,parts,device) for _ in range(args.reps)]
            median=statistics.median(times)
            row={'case':label,'direction':direction,'total_bytes':total,
                 'copy_calls':nparts,'times_ms':times,'median_ms':median,
                 'p95_ms':sorted(times)[int(.95*(len(times)-1))],
                 'effective_gbps':total/1e6/median}
            rows.append(row)
            print(label,direction,round(median,3),'ms',round(row['effective_gbps'],3),'GB/s',flush=True)
        del host,gpu
    result={'scope':'synchronous completion of PyTorch non_blocking copies between pinned CPU DRAM and GPU; no STT-MRAM',
            'config':vars(args),'runtime':runtime_identity(),
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpu':torch.cuda.get_device_name(device),
            'cpu_affinity_count':len(os.sched_getaffinity(0)),
            'pcie_before':before,'pcie_after':link(index),'rows':rows,
            'limitations':'CPU allocation/first touch and GPU NUMA placement are host-specific; no controller, STT device, model overlap, or end-to-end policy.'}
    atomic_json(args.output,result)
    print(json.dumps({'pcie_after':result['pcie_after'],'cases':len(rows)},indent=2))


if __name__=='__main__':main()
