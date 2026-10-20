"""Run remaining frozen untimed GPU jobs serially; preserve every output."""
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
    jobs = {}
    commands = [('all_default', 'audit_resources.py', ['--logits', 'all', '--arm', 'ordinary', '--hot', '1024', '--cold', '2048']),
        ('last_default', 'audit_resources.py', ['--logits', 'last', '--arm', 'ordinary', '--hot', '1024', '--cold', '2048'])]
    commands += [(f'generation_{geometry}', 'check_generation.py', ['--manifest',
        f'results/hardware/fused_bounded_pool_20261002/generation_{geometry}/generation_manifest.json'])
        for geometry in ('shared', 'default')]
    for name, script, args in commands:
        output = BASE / (name + '.json')
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(BASE / script), *args, '--output', str(output)]
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
