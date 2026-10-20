"""Compare native controller with original promotion, backup and address rules."""
from pathlib import Path
from types import SimpleNamespace,MethodType
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from native_controller import kernel
from bulk_promotion_stage import plan,rows,resolve

def frame(size=17,attention=None,inclusive=True,floor=0):
    shadow=np.zeros(size,dtype=np.bool_);shadow[::5]=True
    return dict(shadow=shadow,resident=np.arange(size)%3==0,
        streak=np.asarray(np.arange(size)%4,dtype=np.float32),
        attention=attention if attention is not None else [(float(i%3),i) for i in range(size)],
        inclusive=inclusive,floor=floor)

def reference(f,budget,hysteresis=2):
    cache=SimpleNamespace(stt_pos=list(range(len(f['shadow']))),stt_shadow=torch.from_numpy(f['shadow'].copy()),stt_was_vram_resident=torch.from_numpy(f['resident'].copy()),stt_peek_streak=torch.from_numpy(f['streak'].copy()),cfg=SimpleNamespace(inclusive=f['inclusive'],stt_live_floor=f['floor']),_HYSTERESIS_MIN_STREAK=hysteresis)
    cache.resolve_promotion_indices=MethodType(resolve,cache)
    attention=dict((i,a) for a,i in f['attention']);p=SimpleNamespace(_stt_attn={0:dict((i,a) for a,i in f['attention'])},_stt_attn_idx={0:attention},promotion_budget=budget)
    hits,deferred=plan(p,0,cache);selected,floor_deferred=rows(cache,hits) if hits else ([],0)
    return dict(hits=hits,deferred=deferred,rows=selected,floor_deferred=floor_deferred,streak=cache.stt_peek_streak.numpy())

@pytest.mark.parametrize('budget',[0,1,8,32])
@pytest.mark.parametrize('inclusive,floor',[(True,0),(True,10),(True,100),(False,100)])
def test_native_promotion_ties_floor_and_streak_exact(budget,inclusive,floor):
    frames=[frame(inclusive=inclusive,floor=floor),frame(0),frame(attention=[]),frame(attention=[(1.,0)])]
    actual=kernel().promotion(frames,budget,2)
    for f,a in zip(frames,actual):
        expected=reference(f,budget)
        for name in ['hits','deferred','rows','floor_deferred']:assert a[name]==expected[name],name
        assert np.array_equal(a['streak'].view(np.uint32),expected['streak'].view(np.uint32))


def test_native_float32_streak_nonfinite_signedzero_and_boundary():
    f=frame(7,attention=[(np.inf,0),(np.inf,1),(-np.inf,2),(0.,3),(-0.,4),(1.,5),(1.,6)])
    f['shadow'][:]=False;f['resident'][:]=True
    f['streak']=np.array([0.,-0.,np.inf,-np.inf,np.nan,2**24,np.nextafter(np.float32(0),np.float32(1))],dtype=np.float32)
    expected=reference(f,32);a=kernel().promotion([f],32,2)[0]
    for name in ['hits','deferred','rows','floor_deferred']:assert a[name]==expected[name]
    assert np.array_equal(a['streak'].view(np.uint32),expected['streak'].view(np.uint32))

@pytest.mark.parametrize('case',['empty','ties','duplicates','noninclusive'])
def test_backup_classification_matches_last_shadow_and_victim_order(case):
    f=dict(hot=[4,8,10,16],cold=[4,8,4,10],shadow=np.array([True,False,True,True]),victims=[3,2,0,1],inclusive=case!='noninclusive')
    if case=='empty':f.update(hot=[],cold=[],shadow=np.zeros(0,dtype=np.bool_),victims=[])
    if case=='duplicates':f['victims']=[0,2,0,3]
    lookup={p:i for i,p in enumerate(f['cold']) if f['shadow'][i]} if f['inclusive'] else {}
    expected=dict(restored=[lookup[f['hot'][i]] for i in f['victims'] if f['hot'][i] in lookup],paid=[i for i in f['victims'] if f['hot'][i] not in lookup])
    assert kernel().backups([f])[0]==expected


def test_physical_addresses_layer_head_slot_order_and_int32_boundary():
    frames=[dict(heads=8,capacity=1024,base=8192,slots=[9,2,40]),dict(heads=2,capacity=2048,base=16384,slots=[]),dict(heads=1,capacity=1,base=2**31-1,slots=[0])]
    expected=[f['base']+h*f['capacity']+s for f in frames for h in range(f['heads']) for s in f['slots']]
    result=kernel().addresses(frames);assert result.dtype==np.int64 and result.tolist()==expected

@pytest.mark.parametrize('fault',['nan','shape','index','victim','slot','overflow','negative'])
def test_unsupported_metadata_fails_without_input_mutation(fault):
    f=frame();old=f['streak'].copy()
    if fault=='nan':f['attention']=[(float('nan'),1)]
    elif fault=='shape':f['resident']=f['resident'][:-1]
    elif fault=='index':f['attention']=[(1.,-18)]
    with pytest.raises(ValueError):
        if fault in ['nan','shape','index']:kernel().promotion([f],32,2)
        elif fault=='victim':kernel().backups([dict(hot=[1],cold=[1],shadow=np.ones(1,dtype=np.bool_),victims=[1],inclusive=True)])
        else:kernel().addresses([dict(heads=8,capacity=16,base=(2**31 if fault=='overflow' else 0),slots=([-1] if fault=='negative' else [16]))])
    assert np.array_equal(f['streak'].view(np.uint32),old.view(np.uint32))
