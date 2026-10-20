"""Serial real API/CLI GPU integration checks; preserve all evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def main():
    ledger = BASE / 'jobs.json'
    if ledger.exists():
        raise FileExistsError(ledger)
    manifest = json.loads((ROOT / 'results/hardware/fused_bounded_pool_20261002/generation_shared/generation_manifest.json').read_text())
    source = ROOT / manifest['tasks']['qasper']['source_file']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['tasks']['qasper']['source_sha256']
    commands = []
    flags = ['--cpu-bulk-promotion', '--cpu-reuse-visible', '--cpu-fused-compaction',
        '--cpu-bounded-pool', '--cpu-direct-shared', '--cpu-batched-scores', '--real-query-prefill']
    for arm in ('all', 'last'):
        commands.append((f'cli_{arm}', ROOT / 'experiments/evaluate_query_selection.py',
            ['--data-dir', str(source.parent), '--tasks', 'qasper', '--max-samples', '1',
             '--strategy', 'lru', '--tiered-vram-budget', '991', '--tiered-stt-budget', '1090',
             *flags, *(['--prefill-last-logits'] if arm == 'last' else [])]))
    for geometry, hot, cold in (('shared', 991, 1090), ('default', 1024, 2048)):
        commands.append((f'last_{geometry}', BASE / 'audit_resources.py',
            ['--arm', 'ordinary', '--hot', str(hot), '--cold', str(cold)]))
    commands += [(f'generation_{geometry}', BASE / 'check_generation.py', ['--manifest',
        f'results/hardware/fused_bounded_pool_20261002/generation_{geometry}/generation_manifest.json'])
        for geometry in ('shared', 'default')]
    jobs = {}
    for name, script, args in commands:
        output = BASE / (name + '.json')
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(script), *args, '--output', str(output)]
        with (BASE / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            jobs[name] = dict(pid=process.pid, runner_pid=os.getpid(), status='running', command=command)
            ledger.write_text(json.dumps(jobs, indent=2) + '\n')
            print(name, 'started', process.pid, flush=True)
            code = process.wait()
        jobs[name].update(status='finished', exit_code=code)
        ledger.write_text(json.dumps(jobs, indent=2) + '\n')
        print(name, 'finished', code, flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
