"""Research native-query selector with the actual Mistral query projection.

Scoped hooks run selection after q_proj and before the native KV-cache update.
Only the duplicate projection expression differs from PoolQuerySelector math.
"""
import ast,hashlib,inspect,textwrap
import torch
from transformers.models.mistral.modeling_mistral import MistralAttention,rotate_half
from pool_query_selector import PoolQuerySelector


def ranking_factory():
    tree=ast.parse(textwrap.dedent(inspect.getsource(PoolQuerySelector._hook)))
    before=ast.dump(tree,include_attributes=False)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='module.q_proj']
    assert len(calls)==1 and ast.unparse(calls[0].args[0])=='hidden'
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            if node is calls[0]:return ast.Subscript(value=ast.Name(id='options',ctx=ast.Load()),
                slice=ast.Constant(value='_captured_query_projection'),ctx=ast.Load())
            return self.generic_visit(node)
    candidate=Replace().visit(tree);after=ast.dump(candidate,include_attributes=False)
    namespace=dict(torch=torch,rotate_half=rotate_half)
    exec(compile(ast.fix_missing_locations(candidate),'<reuse-query-projection>','exec'),namespace)
    return namespace['_hook'],dict(control=hashlib.sha256(before.encode()).hexdigest(),candidate=hashlib.sha256(after.encode()).hexdigest())


RANKING_FACTORY,PROJECTION_AST=ranking_factory()


class ReusedQuerySelector:
    def __init__(self,model,policy):
        self.policy=policy;self.handles=[];self.pending={};self.ranking={}
        self.selected_layers=self.scanned_rows=self.projection_reuses=0
        self.inside_selection=False
        try:
            for index,layer in enumerate(model.model.layers):
                attention=layer.self_attn
                if type(attention) is not MistralAttention:raise ValueError('Query projection reuse requires native MistralAttention')
                self.ranking[index]=RANKING_FACTORY(self,index)
                self.handles.append(attention.register_forward_pre_hook(self._before(index),with_kwargs=True))
                self.handles.append(attention.q_proj.register_forward_hook(self._projected(index,attention)))
                self.handles.append(attention.register_forward_hook(self._after(index),always_call=True))
        except BaseException:self.close();raise

    def _before(self,index):
        def before(module,args,options):
            if index in self.pending:raise RuntimeError('Nested attention context is unsupported')
            hidden=options.get('hidden_states',args[0] if args else None)
            if hidden is not None and hidden.shape[1]==1 and index in self.policy.layer_caches:
                self.pending[index]=(args,dict(options))
        return before

    def _projected(self,index,attention):
        def projected(module,args,output):
            context=self.pending.pop(index,None)
            if context is None:return
            positional,options=context;options['_captured_query_projection']=output
            self.inside_selection=True
            try:
                self.ranking[index](attention,positional,options)
                self.projection_reuses+=1
            finally:self.inside_selection=False
        return projected

    def _after(self,index):
        def after(module,args,output):self.pending.pop(index,None)
        return after

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear();self.pending.clear();self.ranking.clear();self.inside_selection=False
