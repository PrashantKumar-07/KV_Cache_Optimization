"""Execute unchanged reference metric function ASTs without unused Chinese imports."""
import ast,datetime,hashlib,json,random,re,string,sys
from collections import Counter
from pathlib import Path
from rouge import Rouge
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
import longbench_metrics as ours
reference=ROOT/'sota/snapkv/experiments/LongBench/metrics.py'
names={'normalize_answer','f1_score','qa_f1_score','rouge_score'}
tree=ast.parse(reference.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in selected}==names
namespace=dict(re=re,string=string,Counter=Counter,Rouge=Rouge)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(reference),'exec'),namespace)
rng=random.Random(20261015);vocab=['the','a','an','Answer','yes','no','unanswerable','42','cancer','New','York','x-y','foo.bar','!','', '\n']
rows=[]
for i in range(250):
 a=' '.join(rng.choices(vocab,k=rng.randrange(12)));b=' '.join(rng.choices(vocab,k=rng.randrange(12)))
 for name in ['qa_f1_score','rouge_score']:
  x=getattr(ours,name)(a,b);y=namespace[name](a,b)
  assert abs(x-y)<1e-12,(name,a,b,x,y)
  rows.append({'metric':name,'prediction':a,'answer':b,'ours':x,'reference':y})
report={'verified':True,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'comparisons':len(rows),'metrics':['qa_f1_score','rouge_score'],
 'scope':'All current six tasks use these two metrics. Exact unmodified reference function AST execution excludes irrelevant jieba/fuzzywuzzy imports; no stubs. This checks metric math, not data/prompts/predictions/generalization.',
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [reference,ROOT/'experiments/longbench_metrics.py',Path(__file__)]},'rows':rows}
out=Path(__file__).with_name('reference_equivalence.json');assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print('PASS',len(rows),'reference comparisons')
