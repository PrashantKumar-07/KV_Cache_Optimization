import numpy as np
import pytest
from cold_boundary import live_plan,address_plan

def owners():
 return dict(cold=np.array([[10,11,12,13],[20,21,22,23]],np.int64),shadow=np.array([[False,True,False,False],[True,False,True,False]]),slots=np.array([[2,0,3,1],[1,3,0,2]],np.int32),cold_n=np.array([4,4],np.int32),hot_n=np.array([3,3],np.int32),version=np.array([1,2],np.int64))

def test_live_plan_preserves_ragged_ascending_rows_and_global_count():
 o=owners();before={k:v.copy() for k,v in o.items()};rows,count=live_plan(o,3)
 assert [r.tolist() for r in rows]==[[0,2,3],[1,3]] and count==2
 assert all(np.array_equal(before[k],v) for k,v in o.items())

def test_physical_address_plan_exact_layer_head_selected_order():
 o=owners();a,p=address_plan(o,np.array([[0,2],[1,3]],np.int32),np.array([0,8],np.int64),2,16)
 assert a.tolist()==[2,3,6,7,11,10,15,14]
 assert p.tolist()==[[10,12],[21,23]]

def test_invalid_owner_or_empty_exposure_rejects_without_mutation():
 o=owners()
 for quota in [-1,True,0]:
  with pytest.raises(ValueError):live_plan(o,quota)
 bad=o|dict(cold_n=np.array([4,5],np.int32))
 with pytest.raises(ValueError):live_plan(bad,2)
 assert o['version'].tolist()==[1,2]

def test_addresses_reject_shadow_duplicate_or_out_of_pool():
 o=owners();bases=np.array([0,8],np.int64)
 for selected in [np.array([[0,1],[1,3]],np.int32),np.array([[0,0],[1,3]],np.int32),np.array([[0,4],[1,3]],np.int32)]:
  with pytest.raises(ValueError):address_plan(o,selected,bases,2,16)
 with pytest.raises(ValueError):address_plan(o,np.array([[0,2],[1,3]],np.int32),np.array([0,0],np.int64),2,16)
