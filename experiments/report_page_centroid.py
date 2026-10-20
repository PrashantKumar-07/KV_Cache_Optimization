"""Verify the fixed-trajectory box/L2/centroid page diagnostic."""
import hashlib
import json
from pathlib import Path

from provenance import atomic_json


ROOT = Path(__file__).resolve().parents[1] / 'results/page_recall'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {'results': {}, 'inputs_sha256': {},
              'scope': 'One Mistral 16K prompt, 20 decode steps, 32 layers. '
                       'Offline summaries rebuilt from all cold keys; local rerank uses '
                       'only fast and candidate keys, but no deployable latency/quality claim.'}
    trajectory = None
    for pages in (8, 32):
        report['results'][str(pages)] = {}
        for mode, suffix in (('box', 'box_recheck'), ('l2', 'l2'),
                             ('centroid', 'centroid_local')):
            path = ROOT / f'p16_k{pages}_{suffix}.json'
            doc = json.loads(path.read_text())
            cfg, rows = doc['config'], doc['records']
            assert cfg['page_size'] == 16 and cfg['candidate_pages'] == pages
            assert cfg['page_bound'] == mode and cfg['ctx_len'] == 16384
            assert cfg['decode'] == 20 and len(rows) == 640
            assert len({(r['layer'], r['step']) for r in rows}) == 640
            this_trajectory = [{k: v for k, v in row.items() if not k.startswith('page_')}
                               for row in rows]
            if trajectory is None:
                trajectory = this_trajectory
                prompt_sha256 = cfg['prompt_sha256']
                runtime = cfg['runtime']
            else:
                assert this_trajectory == trajectory
                assert cfg['prompt_sha256'] == prompt_sha256 and cfg['runtime'] == runtime
            mean = lambda key: sum(row[key] for row in rows) / len(rows)
            result = {'oracle_top32_recall': mean('page_oracle_overlap') / mean('selected'),
                      'oracle_mass_fraction': mean('page_selected_mass') / mean('oracle_mass'),
                      'candidate_row_fraction': mean('page_candidate_rows') / mean('live'),
                      'summary_mib_all_layers': mean('page_summary_bytes') * 32 / 2**20}
            if mode == 'centroid':
                result['local_top32_recall'] = (mean('page_local_oracle_overlap') /
                                               mean('selected'))
                result['local_mass_fraction'] = (mean('page_local_selected_mass') /
                                                 mean('oracle_mass'))
            report['results'][str(pages)][mode] = result
            report['inputs_sha256'][path.name] = sha(path)
    atomic_json(ROOT / 'centroid_report.json', report)
    print(json.dumps(report['results'], indent=2))


if __name__ == '__main__':
    main()
