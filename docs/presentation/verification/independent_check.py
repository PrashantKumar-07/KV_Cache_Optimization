"""Independent recheck of plotted values from raw prediction records.

Own LongBench-style QA F1 (re-implemented, not imported), ROUGE-L via the rouge
package, token-id -> text decode with the real tokenizer, and raw counter sums.
"""
import csv, json, re, string, sys
from collections import Counter, defaultdict
from pathlib import Path
from rouge import Rouge
from transformers import AutoTokenizer

ROOT = Path(sys.argv[1])
proof = json.loads((ROOT / 'results/full_evaluation_20261015/verified_results.json').read_text())
tok = AutoTokenizer.from_pretrained(str(ROOT.parent / 'models/mistral-7b-instruct'))

def norm(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1(pred, gt):
    p, g = norm(pred).split(), norm(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    pr, rc = same / len(p), same / len(g)
    return 2 * pr * rc / (pr + rc)

R = Rouge()
def rl(pred, gt):
    try:
        return R.get_scores([pred], [gt], avg=True)['rouge-l']['f']
    except Exception:
        return 0.0

agg = defaultdict(lambda: defaultdict(float))
n = defaultdict(int)
mismatch_score = mismatch_decode = 0
worst = 0.0
for fname in proof['record_sha256']:
    r = json.loads((ROOT / fname).read_text())
    metric = rl if r['metric_function'] == 'rouge_score' else f1
    mine = 100 * max(metric(r['scored_prediction'], a) for a in r['answers'])
    d = abs(mine - r['score'])
    worst = max(worst, d)
    if d > 1e-6:
        mismatch_score += 1
    dec = tok.decode(r['generated_token_ids'], skip_special_tokens=True).strip()
    if dec != r['prediction'].strip():
        mismatch_decode += 1
    k = (r['task'], r['arm'])
    n[k] += 1
    agg[k]['score'] += mine
    for c in ('paid_writes', 'writes_saved', 'demoted', 'promoted', 'dropped'):
        agg[k][c] += r['cache_stats'].get(c, 0)

print(f'records={sum(n.values())} rescore_mismatch(>1e-6)={mismatch_score} max_abs_diff={worst:.3g} decode_text_mismatch={mismatch_decode}')
plotted = {(row['task'], row['arm']): row for row in csv.DictReader(open(ROOT / 'results/figures_20261118_renumbered/task_values.csv'))}
bad = 0
for k in sorted(n):
    mean = agg[k]['score'] / n[k]
    p = plotted[k]
    ds = abs(mean - float(p['score']))
    dw = abs(agg[k]['paid_writes'] - float(p['paid_writes'])) + abs(agg[k]['writes_saved'] - float(p['writes_saved']))
    flag = 'OK' if ds < 1e-6 and dw == 0 and n[k] == int(p['n']) else 'DIFF'
    bad += flag != 'OK'
    print(f'{flag} {k[0]:16s} {k[1]:15s} n={n[k]:3d} my_mean={mean:7.3f} plotted={float(p["score"]):7.3f} paid={int(agg[k]["paid_writes"]):>10,} saved={int(agg[k]["writes_saved"]):>11,}')
print('rows differing from plotted table:', bad)
