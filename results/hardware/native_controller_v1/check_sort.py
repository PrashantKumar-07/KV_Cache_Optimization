"""GPU grouped unstable-sort equivalence probe; no model or adoption claim."""
import hashlib,json,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent
def main():
    output=BASE/'sort.json';assert not output.exists();rows=[];torch.manual_seed(0)
    for size in [1,17,32,257,850,897,991,1024,1090,2048,4096]:
        for case in ['all_ties','timestamps','random','nonfinite']:
            tensors=[]
            for layer in range(32):
                t=torch.zeros(size,device='cuda') if case=='all_ties' else torch.randint(0,51,(size,),device='cuda').float()
                if case=='random':t=torch.randn(size,device='cuda')
                if case=='nonfinite' and size>=4:t[:4]=torch.tensor([float('nan'),float('inf'),-float('inf'),0.],device='cuda')
                tensors.append(t)
            expected=torch.stack([torch.argsort(t,descending=True) for t in tensors])
            actual=torch.argsort(torch.stack(tensors),dim=1,descending=True)
            rows.append(dict(size=size,case=case,exact=torch.equal(actual,expected),different_rows=int((actual!=expected).any(dim=1).sum())))
    report=dict(passed=all(r['exact'] for r in rows),rows=rows,torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Grouped vs separate CUDA unstable descending argsort at identical unpadded row lengths. No fullmodel/retention/timing proof; failures forbid assuming grouped sort preserves old tie order.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],cases=len(rows),failures=[r for r in rows if not r['exact']])))
if __name__=='__main__':main()
