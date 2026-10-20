"""Quiet-GPU preflight, monitored serial ABBA guard jobs, no retry."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def sample(allowed=()):
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
        pids = [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]
        return dict(monotonic=time.monotonic(), pids=pids, unexpected=[pid for pid in pids if pid not in allowed])
    except Exception as error:
        return dict(monotonic=time.monotonic(), error=str(error), unexpected=['observation_failed'])


def guards(raw):
    summary = {}
    for arm in ('fused_bounded', 'native_reuse'):
        trials = [trial for row in raw['rows'] if row['arm'] == arm for trial in row['trials']]
        steps = [value for trial in trials for value in trial['step_ms']]
        summary[arm] = dict(median_ms=statistics.median(steps), p95_ms=float(np.percentile(steps, 95)),
            ttft_ms=statistics.median(trial['ttft_ms'] for trial in trials),
            prefill_peak_gb=max(trial['prefill_peak_memory_gb'] for trial in trials),
            decode_peak_gb=max(trial['decode_peak_memory_gb'] for trial in trials))
    control, candidate = summary['fused_bounded'], summary['native_reuse']
    checks = {name: candidate[name] <= factor * control[name] for name, factor in
        (('median_ms', 1.02), ('p95_ms', 1.02), ('ttft_ms', 1.10), ('prefill_peak_gb', 1.02), ('decode_peak_gb', 1.02))}
    return dict(summary=summary, checks=checks, passed=all(checks.values()))


def main():
    ledger = BASE / 'timing_jobs.json'
    if ledger.exists():
        raise FileExistsError(ledger)
    samples = []
    for _ in range(20):
        samples.append(sample())
        time.sleep(2)
    preflight = dict(samples=samples, quiet=all(not row['unexpected'] for row in samples))
    (BASE / 'timing_preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    jobs = {}
    if not preflight['quiet']:
        ledger.write_text(json.dumps(dict(status='preflight_rejected', jobs=jobs), indent=2) + '\n')
        print('Timing not started: quiet GPU preflight failed.', flush=True)
        return
    for geometry, hot, cold in (('shared', 991, 1090), ('default', 1024, 2048)):
        output = BASE / f'timing_{geometry}.json'
        if output.exists():
            raise FileExistsError(output)
        command = [sys.executable, str(BASE / 'compare_timing.py'), '--contexts', '16384',
            '--decode', '50', '--warmup', '1', '--blocks', '3', '--hot', str(hot),
            '--cold', str(cold), '--output', str(output)]
        observed = []; stop = threading.Event()
        with (BASE / f'timing_{geometry}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            jobs[geometry] = dict(pid=process.pid, runner_pid=os.getpid(), status='running', command=command)
            ledger.write_text(json.dumps(dict(status='running', jobs=jobs), indent=2) + '\n')
            print(geometry, 'timing started', process.pid, flush=True)

            def observe():
                while not stop.is_set():
                    observed.append(sample((process.pid,)))
                    stop.wait(2)

            thread = threading.Thread(target=observe, daemon=True); thread.start()
            try:
                code = process.wait()
            finally:
                stop.set(); thread.join()
        monitor = dict(samples=observed, clean=bool(observed) and all(not row['unexpected'] for row in observed))
        (BASE / f'timing_monitor_{geometry}.json').write_text(json.dumps(monitor, indent=2) + '\n')
        jobs[geometry].update(status='finished', exit_code=code, clean_gpu=monitor['clean'])
        ledger.write_text(json.dumps(dict(status='running', jobs=jobs), indent=2) + '\n')
        print(geometry, 'timing finished', code, 'clean', monitor['clean'], flush=True)
        if code or not monitor['clean']:
            break
        decision = guards(json.loads(output.read_text()))
        (BASE / f'timing_screen_{geometry}.json').write_text(json.dumps(decision, indent=2) + '\n')
        if not decision['passed']:
            print('Guard failed; stop timing expansion.', flush=True)
            break
    ledger.write_text(json.dumps(dict(status='finished', jobs=jobs), indent=2) + '\n')


if __name__ == '__main__':
    main()
