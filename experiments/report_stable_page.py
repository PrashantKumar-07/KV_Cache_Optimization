"""Verify position versus stable-dense centroid grouping on one trajectory."""
import hashlib
import json
from pathlib import Path

from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1] / 'results/page_recall'


def load(name):
    path = ROOT / name
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(doc):
    rows = doc['records']
    mean = lambda k: sum(row[k] for row in rows) / len(rows)
    s = {'candidate_rows': mean('page_candidate_rows'),
         'candidate_fraction': mean('page_candidate_rows') / mean('live'),
         'top32_recall': mean('page_local_oracle_overlap') / mean('selected'),
         'oracle_mass_fraction': mean('page_local_selected_mass') / mean('oracle_mass'),
         'summary_mib_all_layers': mean('page_summary_bytes') * 32 / 2**20}
    if 'page_count' in rows[0]:
        s['active_pages'] = mean('page_count')
    if 'page_underfilled' in rows[0]:
        s['underfilled_layer_step_fraction'] = mean('page_underfilled')
    return s


def main():
    out = {'inputs_sha256': {}, 'by_budget': {},
           'scope': 'One Mistral 16K prompt, 20 decode steps x 32 layers; '
                    'offline full-key centroid rebuild, no serving latency or physical read claim.'}
    for budget in (8, 32):
        names = {'row': f'p16_k{budget}_centroid_local.json',
                 'absolute_position': f'p16_k{budget}_position_centroid.json',
                 'stable_dense': f'p16_k{budget}_stable_dense_v2.json'}
        docs = {}
        for mode, name in names.items():
            docs[mode], out['inputs_sha256'][name] = load(name)
            cfg, rows = docs[mode]['config'], docs[mode]['records']
            assert cfg['ctx_len'] == 16384 and cfg['decode'] == 20
            assert cfg['page_size'] == 16 and cfg['candidate_pages'] == budget
            assert len(rows) == len({(r['layer'], r['step']) for r in rows}) == 640
        base = docs['row']
        for other in (docs['absolute_position'], docs['stable_dense']):
            assert other['config']['prompt_sha256'] == base['config']['prompt_sha256']
            assert other['config']['runtime'] == base['config']['runtime']
            for a, b in zip(base['records'], other['records']):
                for field in ('layer', 'step', 'live', 'selected', 'oracle_overlap',
                              'slow_mass', 'selected_mass', 'oracle_mass'):
                    assert a[field] == b[field], (budget, field)
        out['by_budget'][str(budget)] = {mode: metrics(doc) for mode, doc in docs.items()}
        assert out['by_budget'][str(budget)]['stable_dense']['underfilled_layer_step_fraction'] == 0
    atomic_json(ROOT / 'stable_page_report.json', out)
    print(json.dumps(out['by_budget'], indent=2))


if __name__ == '__main__':
    main()
