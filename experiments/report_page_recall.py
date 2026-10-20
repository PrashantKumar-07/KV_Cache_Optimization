"""Validate paired offline page probes and summarize recall/traffic proxies."""
import hashlib
import json
from pathlib import Path
import numpy as np
from provenance import atomic_json, runtime_identity


def main():
    root = Path('results/page_recall')
    previous = None
    summary = {}
    for name in ('p16_k8', 'p16_k32'):
        path = root / f'{name}.json'
        doc = json.loads(path.read_text())
        c, rows = doc['config'], doc['records']
        assert c['runtime'] == runtime_identity()
        for key, file in [('probe_sha256', 'probe_query_recall.py'),
                          ('page_selector_sha256', 'page_candidates.py')]:
            assert c[key] == hashlib.sha256(Path('experiments', file).read_bytes()).hexdigest()
        assert len(rows) == c['decode'] * 32
        assert len({(r['layer'], r['step']) for r in rows}) == len(rows)
        trajectory = [{k:v for k,v in r.items() if not k.startswith('page_')} for r in rows]
        if previous is not None:
            assert trajectory == previous, 'Different decode trajectory'
        previous = trajectory
        for r in rows:
            assert 0 <= r['page_oracle_overlap'] <= r['selected']
            assert r['page_candidate_rows'] <= c['page_size'] * c['candidate_pages']
            assert r['page_selected_mass'] <= r['oracle_mass'] + 1e-7
        mean = lambda key: float(np.mean([r[key] for r in rows]))
        summary[name] = {
            'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'top32_recall': mean('page_oracle_overlap') / mean('selected'),
            'oracle_mass_fraction': mean('page_selected_mass') / mean('oracle_mass'),
            'candidate_key_fraction': mean('page_candidate_rows') / mean('live'),
            'summary_bytes_all_layers': mean('page_summary_bytes') * 32,
            'candidate_key_bytes_all_layers': mean('page_candidate_rows') * 2048 * 32}
    atomic_json(root / 'report.json', {'results':summary,
        'scope':'One fixed Mistral 16k prompt, 20 steps, identical access-time trajectory. '
                'Offline full-key reranking; candidate bytes exclude metadata maintenance, values and transfers. '
                'No quality or latency claim.'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
