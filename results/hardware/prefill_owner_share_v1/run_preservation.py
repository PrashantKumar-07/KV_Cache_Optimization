"""Run queued preservation jobs once, serially under the caller's GPU lock."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def main():
    shared = json.loads((BASE / 'generation_shared.json').read_text())
    assert shared['passed'] and len(shared['rows']) == 24, 'Shared generation must pass first'
    jobs_path = BASE / 'jobs.json'
    jobs = json.loads(jobs_path.read_text())
    jobs['shared_generation'].update(status='finished', exit_code=0)
    jobs['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    commands = [('generation_default', ['experiments/check_prefill_owner_generation.py',
        '--manifest', 'results/hardware/fused_bounded_pool_20261002/generation_default/generation_manifest.json'])]
    for kind in ('boundary', 'interleaved'):
        for geometry in ('shared', 'default'):
            flags = ['--default-geometry'] if geometry == 'default' else []
            commands.append((kind + '_' + geometry,
                [str(BASE / 'check_preservation.py'), '--kind', kind, *flags]))
    for name, arguments in commands:
        output = BASE / (name + '.json'); log = BASE / (name + '.log')
        if output.exists() or log.exists():
            raise FileExistsError(f'Preserve prior job evidence: {name}')
        jobs[name] = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
            command=[sys.executable, *arguments, '--output', str(output)])
        jobs_path.write_text(json.dumps(jobs, indent=2) + '\n')
        print(name, 'started', flush=True)
        with log.open('x') as stream:
            result = subprocess.run(jobs[name]['command'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        passed = output.exists() and json.loads(output.read_text()).get('passed', False)
        jobs[name].update(status='finished' if result.returncode == 0 and passed else 'failed',
            exit_code=result.returncode, ended=datetime.now(timezone.utc).isoformat())
        jobs_path.write_text(json.dumps(jobs, indent=2) + '\n')
        print(name, jobs[name]['status'], flush=True)
        assert result.returncode == 0 and passed, f'{name} failed; stop advancement'


if __name__ == '__main__':
    main()
