"""Prove the 2026-11-18 renumbering changed names only, never figure content.

Checks, without trusting any renderer:
1. every regenerated output here is byte-identical to the October output of the
   same figure (old name), whose hash is in the October deployment records;
2. every file in figures_final/ is byte-identical to the regenerated output;
3. the legacy plotting scripts removed from experiments/ are preserved
   byte-exact in the frozen evaluation source tarball, matching plan.json.
"""
from pathlib import Path
import hashlib, json, re, tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
OLD_TO_NEW = {
    'fig1_accuracy_equivram': 'fig1_accuracy_equivram', 'fig7_accuracy_fair': 'fig2_accuracy_fair',
    'fig3_write_savings': 'fig3_write_savings', 'fig11_footprint': 'fig4_footprint',
    'fig4_decode_scaling': 'fig5_decode_scaling', 'fig2_latency_comparison': 'fig6_modeled_traffic_time',
    'fig12_energy': 'fig7_energy', 'fig6_migration_profile': 'fig8_migration_profile',
    'fig14_heatmap': 'fig9_heatmap', 'supplement_faithful_accuracy': 'supplement_faithful_accuracy'}
OCT = ROOT / 'results/audit_20261016_current_figures'
OCT_T = ROOT / 'results/audit_20261016_timing/figures'
deployed = set(re.findall(r'[0-9a-f]{64}', (OCT / 'deployment.json').read_text() + (OCT_T / 'deployment.json').read_text()))
rows = []
for old, new in OLD_TO_NEW.items():
    for ext in ('pdf', 'png'):
        old_path = (OCT_T if old == 'fig4_decode_scaling' else OCT) / f'{old}.{ext}'
        final = ROOT / 'figures_final' / ('supplemental/' if new.startswith('supplement') else '') / f'{new}.{ext}'
        now, before, shipped = (sha(p.read_bytes()) for p in (HERE / f'{new}.{ext}', old_path, final))
        row = dict(old=old_path.relative_to(ROOT).as_posix(), new=final.relative_to(ROOT).as_posix(), sha256=now,
                   regenerated_equals_october=now == before, final_equals_regenerated=shipped == now,
                   october_hash_in_deployment_record=before in deployed)
        assert row['regenerated_equals_october'] and row['final_equals_regenerated'] and row['october_hash_in_deployment_record'], row
        rows.append(row)
plan = json.loads((ROOT / 'results/full_evaluation_20261015/plan.json').read_text())['source_sha256']
removed = ['experiments/plot_figures_v2.py', 'experiments/plot_publication_figures.py', 'experiments/plot_frac_ablation.py',
           'experiments/plot_frac_compare.py', 'experiments/archive_plot_all_figures.py']
preserved = {}
with tarfile.open(ROOT / 'results/full_evaluation_20261015/source.tar.gz') as tar:
    for name in removed:
        digest = sha(tar.extractfile(name).read())
        assert digest == plan[name], name
        preserved[name] = digest
result = {'verified': True, 'outputs_checked': len(rows), 'outputs': rows,
          'legacy_scripts_preserved_in_source_tarball': preserved, 'old_to_new': OLD_TO_NEW,
          'proof_sha256': sha(Path(__file__).read_bytes())}
(HERE / 'renumbering_proof.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'verified': True, 'outputs_checked': len(rows), 'legacy_scripts_preserved': len(preserved)}))
