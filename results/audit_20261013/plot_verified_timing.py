"""Plot separate frozen timing studies from raw rows; never combine scopes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/audit_20261013/figures')
assert not (OUT/'manifest.json').exists(), 'Preserve prior figure pack'
OUT.mkdir(exist_ok=True)
studies = [
 ('Faithful SnapKV comparison\n(earlier CPU configuration)', 'early_shared_cpu_phase_v1', 'snap', 'cpu'),
 ('Internal bulk controller\n(frozen earlier source)', 'bulk_payload_controller_v1', 'control', 'query'),
 ('Internal cold-boundary controller\n(frozen before audit fixes)', 'cold_boundary_controller_v1', 'control', 'query')]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
manifest = {'scope': 'Three separate measured fixed-prompt timing studies, not one joint quality/resource comparison. Old-source evidence is historical; no SOTA claim.',
            'units': 'ms/token', 'statistics': 'Pooled per-token median and p95; faint connected points are individual ABBA block medians, not confidence intervals.',
            'source_sha256': {}, 'studies': []}
for ax, (title, folder, first, second) in zip(axes, studies):
    raw = Path('results/hardware')/folder/'cost.json'
    proof = raw.with_name('verified_cost.json')
    data = json.loads(raw.read_text())
    assert data['passed'] and len(data['rows']) == 12
    manifest['source_sha256'].update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw, proof]})
    arms = [first, second]
    summary = {}
    for i, arm in enumerate(arms):
        rows = [row for row in data['rows'] if row['arm'] == arm]
        if not rows:
            raise ValueError((raw, arm, sorted({row['arm'] for row in data['rows']})))
        values = [value for row in rows for trial in row.get('trials', [row]) for value in trial['step_ms']]
        summary[arm] = {'measured_forwards': len(values), 'median': float(np.median(values)), 'p95': float(np.quantile(values, .95))}
        ax.bar(i, summary[arm]['median'], color=['#607d8b', '#16857b'][i], width=.55)
        ax.scatter(i, summary[arm]['p95'], marker='_', color='black', s=180, label='p95' if i == 0 else None)
        ax.text(i, summary[arm]['median']+1, f"{summary[arm]['median']:.2f}", ha='center', fontsize=9)
    for block in range(3):
        medians = [np.median([v for row in data['rows'] if row['block'] == block and row['arm'] == arm for trial in row.get('trials', [row]) for v in trial['step_ms']]) for arm in arms]
        ax.plot(range(2), medians, marker='o', linewidth=.7, color='black', alpha=.22)
    labels = ['SnapKV adapter', 'CPU policy'] if folder.startswith('early') else ['Own control', 'Own candidate']
    ax.set_xticks(range(2), labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=.15)
    manifest['studies'].append({'folder': folder, 'summary': summary})
axes[0].set_ylabel('Complete decode-loop time (ms/token)')
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
for extension in ['png', 'pdf']:
    fig.savefig(OUT/f'frozen_timing_scopes.{extension}', dpi=180, bbox_inches='tight')
plt.close(fig)
manifest['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()}
(OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
