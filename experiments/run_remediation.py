#!/usr/bin/env python3
"""Run the review's bounded reproducibility matrix on available local GPUs.

Jobs checkpoint through the evaluators. Status and logs survive an interrupted
interactive session. No old result files are overwritten. Reinvoke with the
same arguments to resume; runtime mismatches fail rather than blend results.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from provenance import atomic_json, read_jsonl, runtime_identity

ROOT = Path(__file__).resolve().parents[1]
SHORT = ['multifieldqa_en', 'qasper', 'narrativeqa', 'hotpotqa', 'triviaqa']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='../models/mistral-7b-instruct')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--outdir', default='results/remediation')
    p.add_argument('--gpus', nargs='+', default=['0', '1'])
    p.add_argument('--samples', type=int, default=25)
    p.add_argument('--gov-samples', type=int, default=10)
    args = p.parse_args()
    os.chdir(ROOT)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    identity = runtime_identity()
    status_path = out / 'jobs.json'
    status = {'runtime': identity, 'jobs': {}}
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status['runtime'] != identity:
            raise ValueError('Runtime changed; use a new outdir')
    jobs = []
    common = [sys.executable, '-u', 'experiments/longbench_eval.py',
              '--model', args.model, '--data-dir', args.data_dir,
              '--budget', '1024', '--seed', '0', '--max-ctx', '31500',
              '--tiered-kv-dtype', 'bfloat16', '--sttram-bifurcation-frac', '.75']

    def add_eval(name, tasks, samples, vram=1024, stt=2048, methods=None, extra=()):
        jobs.append((name, common + ['--tasks', *tasks, '--methods',
            *(methods or ['tieredkv']), '--max-samples', str(samples),
            '--tiered-vram-budget', str(vram), '--tiered-stt-budget', str(stt),
            '--json', str(out / (name+'.json')), *extra]))

    all_methods = ['full', 'streamingllm', 'h2o', 'snapkv', 'tieredkv']
    add_eval('equal_vram_short', SHORT, args.samples, methods=all_methods)
    add_eval('equal_vram_gov', ['gov_report'], args.gov_samples, methods=all_methods)
    for label, vram, stt in [('equal_attended', 992, 2048), ('equal_total', 341, 683)]:
        add_eval(label+'_short', SHORT, args.samples, vram, stt)
        add_eval(label+'_gov', ['gov_report'], args.gov_samples, vram, stt)
    for weight in ['0.01', '0.05']:
        add_eval('lambda_'+weight, SHORT, args.samples, extra=['--write-aware-lambda',weight])
    for window in [32,64,128,256]:
        add_eval('window_'+str(window), SHORT, args.samples, extra=['--recent-size',str(window)])
    # A real long document, fixed across methods and repeats.
    rows = read_jsonl(Path(args.data_dir)/'gov_report_e.jsonl')
    prompt_path = out / 'latency_prompt.txt'
    prompt_path.write_text(max((r['context'] for r in rows), key=len))
    jobs.insert(0, ('latency', [sys.executable, '-u', 'experiments/latency_profile.py',
        '--model', args.model, '--prompt-file', str(prompt_path), '--output', str(out/'latency.json')]))
    jobs.append(('ppl_validation_budget256', [sys.executable, '-u', 'experiments/ppl_eval.py',
        '--model',args.model,'--texts-jsonl',str(Path(args.data_dir)/'gov_report_e.jsonl'),
        '--n-docs','2','--budget','256','--max-len','1024',
        '--checkpoints','256','512','1024',
        '--output',str(out/'ppl_validation_budget256.json')]))
    lock = threading.Lock()

    def worker(gpu, assigned):
        for name, command in assigned:
            with lock:
                prior = status['jobs'].get(name, {})
                if (prior.get('state') == 'complete' and prior.get('command') == command
                        and (out / (name + '.json')).is_file()):
                    continue
                status['jobs'][name] = {'state':'running','gpu':gpu,'command':command}
                atomic_json(status_path,status)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1',
                       HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (out/(name+'.log')).open('a') as stream:
                code = subprocess.call(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with lock:
                status['jobs'][name].update(state='complete' if code==0 else 'failed',exit_code=code)
                atomic_json(status_path,status)
                print(name,status['jobs'][name]['state'],flush=True)
            if code:
                raise RuntimeError(f'{name} failed; inspect {out/name}.log')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker,gpu,jobs[i::len(args.gpus)]) for i,gpu in enumerate(args.gpus)]
        for f in futures:
            f.result()


if __name__ == '__main__':
    main()
