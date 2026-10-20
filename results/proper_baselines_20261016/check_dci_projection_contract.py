"""CPU correctness proof using the compiled official index, not quality evidence.

Compare native C projection with head-correct precomputed projection. Axis
projection uses exactly representable BF16 values to isolate layout from GEMM
rounding. Debug/serial mode fixes the upstream index RNG for this test only.
"""
from pathlib import Path
import hashlib,json
import torch,dci

BASE=Path(__file__).resolve().parent

def build(keys,projection,provided):
    heads,n,dim=keys.shape
    db=dci.PyDCI(dim,1,1,promotion_prob=.1,promotion_prob_subseq=.1,
        max_volume=512,transform=True,num_inst=heads,parallel_level=0,
        debug=True,proj_vec_t=projection)
    db.add_query_torch(keys,torch.empty(0,dtype=torch.bfloat16),keys.clone(),
        torch.ones(heads,n,dtype=torch.bool),num_levels=-100,num_neighbours=12,
        blind=False,c_num_to_visit=n,q_num_to_visit=n,c_num_to_retrieve=-1,
        q_num_to_retrieve=-1,c_prop_to_visit=1.,q_prop_to_visit=1.,
        c_prop_to_retrieve=1.,q_prop_to_retrieve=.6,c_field_of_view=30,
        q_field_of_view=30,max_num_points=n,transform=True,parallel_level=0,
        attention_mask=torch.full((n,),n,dtype=torch.int32),random=False,
        do_query=False,track=False,update_addr=False,
        changed_page_list=torch.empty(0,dtype=torch.int32),
        data_proj_all=provided.clone(),interval=60,X=60,anchor_threshold=1.02)
    # The API reports actual token -> (leaf, offset) assignments and point counts.
    mapping=db.get_token2node()
    assert all(torch.is_tensor(x) for x in mapping)
    assert torch.equal(db.get_num_points(),torch.full((heads,),n,dtype=torch.int32))
    return tuple(x.clone() for x in mapping),db.get_num_leaves().clone()

def main():
    output=BASE/'dci_projection_contract.json';assert not output.exists()
    torch.set_num_threads(1);torch.manual_seed(0)
    report={'verified':False,'cases':[],'GPU_or_quality_proof':False,
        'scope':'Official compiled index: exact-axis synthetic projection contract; debug serial mode is only a deterministic correctness test, not benchmark configuration.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'compiled_dci_sha256':hashlib.sha256(Path(dci.__file__).read_bytes()).hexdigest()}
    try:
        for heads in (1,2,8):
            n,dim=192,128
            sequence=torch.randint(-64,65,(1,256,heads,dim)).bfloat16()
            keys=torch.stack([sequence[0,32:224,h] for h in range(heads)]).contiguous()
            projection=torch.zeros(dim+1,dtype=torch.bfloat16);projection[0]=1
            correct=(sequence.transpose(1,2).reshape(heads,-1,dim)[:,32:224]@projection[:-1]).float().reshape(heads,n,1).contiguous()
            raw=(sequence.reshape(heads,-1,dim)[:,32:224]@projection[:-1]).float().reshape(heads,n,1).contiguous()
            assert torch.equal(correct[:,:,0],keys[:,:,0].float())
            native=build(keys,projection,torch.empty(0))
            corrected=build(keys,projection,correct)
            original=build(keys,projection,raw)
            same=lambda a,b:all(torch.equal(x,y) for x,y in zip(a[0],b[0])) and torch.equal(a[1],b[1])
            assert same(native,corrected),'Corrected projection disagrees with native C projection'
            report['cases'].append({'KVheads':heads,'indexed_points_per_head':n,
                'corrected_matches_native_token_leaf_offset_and_leaf_counts':True,
                'original_matches_native':same(native,original),
                'original_projection_mismatched_rows':int((raw!=correct).any(-1).sum())})
        report['verified']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';raise
    finally:output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
