"""Original versus projection-reused visible-only no-op hook3ABBA diagnostic."""
import ast,hashlib,importlib.util,inspect,json,sys,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from reused_query_projection import ReusedQuerySelector
from compare_cpu_batched import digest
BASE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    old=load('original_visible_query_cost',ROOT/'results/hardware/query_hook_cost_v1/check_timing.py')
    tree=ast.parse(textwrap.dedent(inspect.getsource(old.VisibleQueryMath._hook)))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='module.q_proj'];assert len(nodes)==1
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            if node is nodes[0]:return ast.Subscript(value=ast.Name(id='options',ctx=ast.Load()),
                slice=ast.Constant(value='_captured_query_projection'),ctx=ast.Load())
            return self.generic_visit(node)
    namespace=old.__dict__.copy()
    exec(compile(ast.fix_missing_locations(Replace().visit(tree)),'<reused-visible-query-math>','exec'),namespace)
    factory=namespace['_hook']
    class ReusedVisibleQueryMath(ReusedQuerySelector):
        def __init__(self,model,policy):
            super().__init__(model,policy)
            self.ranking={i:factory(self,i) for i in range(len(model.model.layers))}
    template=ROOT/'results/hardware/pool_query_selector_v1/check_timing.py'
    source=template.read_text()
    target="selector=PoolQuerySelector(model,policy) if arm=='query' else None"
    assert source.count(target)==1
    source=source.replace(target,"selector=(ReusedVisibleQueryMath if arm=='query' else VisibleQueryMath)(model,policy)")
    source=source.replace("base/'timing_plan.json'","base/'cost_plan.json'").replace("base/'timing.json'","base/'cost.json'")
    scope=dict(__name__='scoped_visible_reuse_timing',__file__=str(Path(__file__)))
    exec(compile(source,'<unchanged-timing-with-scoped-selector>', 'exec'),scope)
    scope.update(ReusedVisibleQueryMath=ReusedVisibleQueryMath,VisibleQueryMath=old.VisibleQueryMath)
    original_profile=scope['profile_one']
    plan=json.loads((BASE/'cost_plan.json').read_text());ref=json.loads((ROOT/plan['control']).read_text())
    assert hashlib.sha256((ROOT/plan['control']).read_bytes()).hexdigest()==plan['control_sha256']
    def profile(model,tokenizer,policy,*args,**options):
        result=original_profile(model,tokenizer,policy,*args,**options)
        state=digest(policy.layer_caches);counters=policy._live_totals()
        assert state==ref['state_sha256'] and counters==ref['counters']
        assert result['trials'][0]['predicted_token_ids']==ref['predicted_token_ids']
        for field in ('host_read_bytes','host_write_bytes'):
            assert sum(getattr(c,field) for c in policy.layer_caches.values())==ref['prototype_counts'][field]
        result.update(state_sha256=state,logical_counters=counters,ordinary_state_transfers_exact=True)
        return result
    scope['profile_one']=profile
    scope['main']()
if __name__=='__main__':main()
