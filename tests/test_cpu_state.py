"""Native CPU phase invariants: roots, commands, FIFO slots and atomic rejection."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from cpu_state import CpuStateArena,kernel


def fixture(layers=2):
    s=CpuStateArena(layers,8,7)
    for layer in range(layers):
        s.load(layer,[10,20,30],[10,50],[False,True,False],[True,False],[True,False],np.array([2.,3.],np.float32),[4,1],[0,2,3,5,6])
    return s


def packed(values,width=None,dtype=np.int32):
    width=max(map(len,values),default=0) if width is None else width
    result=np.full((len(values),width),-1,dtype=dtype)
    for i,v in enumerate(values):result[i,:len(v)]=v
    return result,np.asarray([len(v) for v in values],dtype=np.int32)


def test_whole_demotion_retains_owners_and_emits_complete_commands():
    s=fixture();roots={k:a.ctypes.data for k,a in s.owners.items()}
    commands=s.demote(*packed([[0,2],[]]));s.validate_commands(commands)
    assert commands['sizes'].tolist()==[[1,1,1],[0,0,3]]
    assert commands['paid'][0,:1].tolist()==[2] and commands['restored'][0,:1].tolist()==[0]
    assert commands['writes'][0,:1].tolist()==[0] and commands['keep'][0,:1].tolist()==[1]
    assert s.view('hot',0).tolist()==[20] and s.view('grace',0).tolist()==[False]
    assert s.view('cold',0).tolist()==[10,50,30] and s.view('shadow',0).tolist()==[False]*3
    assert s.view('resident',0).tolist()==[True,False,True] and s.view('streak',0).tolist()==[0,3,0]
    assert s.view('slots',0).tolist()==[4,1,0] and s.view('free',0).tolist()==[2,3,5,6]
    assert roots=={k:a.ctypes.data for k,a in s.owners.items()}
    assert all(not a.flags.writeable for a in commands.values() if isinstance(a,np.ndarray))
    s.append_hot(*packed([[80],[]],dtype=np.int64))
    with pytest.raises(ValueError,match='Stale'):s.validate_commands(commands)


def test_duplicate_cold_positions_restore_last_shadow_only():
    s=CpuStateArena(1,3,4)
    s.load(0,[9],[9,9],[False],[True,True],[False,True],np.array([1.,2.],np.float32),[2,0],[1,3])
    cmd=s.demote(*packed([[0]]))
    assert cmd['restored'][0,0]==1 and s.view('shadow',0).tolist()==[True,False]
    assert s.view('streak',0).tolist()==[1,0]


def test_exclusive_demotion_ignores_stored_shadow_flags():
    s=fixture(1);cmd=s.demote(*packed([[0]]),inclusive=False)
    assert cmd['sizes'][0,:2].tolist()==[1,0] and s.view('cold',0).tolist()==[10,50,10]


def test_reclaim_reorders_without_alias_corruption_and_preserves_f32_bits():
    s=CpuStateArena(1,4,6)
    bits=np.array([0x80000000,0x7fc01234,0x7f801234,1],np.uint32)
    s.load(0,[],[10,20,30,40],[],[True,False,True,False],[False,True,True,False],bits.view(np.float32),[3,0,5,1],[2,4])
    roots={k:a.ctypes.data for k,a in s.owners.items()}
    s.reclaim(*packed([[3,0,2]]))
    assert s.view('cold',0).tolist()==[40,10,30]
    assert s.view('streak',0).view(np.uint32).tolist()==bits[[3,0,2]].tolist()
    assert s.view('free',0).tolist()==[2,4,0]
    assert roots=={k:a.ctypes.data for k,a in s.owners.items()}

@pytest.mark.parametrize('inclusive',[True,False])
def test_promotions_preserve_order_grace_slots_and_live_floor(inclusive):
    s=CpuStateArena(1,6,5)
    s.load(0,[0],[11,12,13],[False],[False]*3,[False,True,False],np.array([1,2,3],np.float32),[4,2,0],[1,3])
    commands=s.promote(*packed([[2,0]]),inclusive=inclusive,floor=1);s.validate_commands(commands)
    assert commands['read_slots'][0].tolist()==[0,4]
    assert s.view('hot',0).tolist()==[0,13,11] and s.view('grace',0).tolist()==[False,True,True]
    if inclusive:
        assert s.view('shadow',0).tolist()==[True,False,True] and s.view('resident',0).tolist()==[True]*3
        assert s.view('free',0).tolist()==[1,3]
    else:
        assert s.view('cold',0).tolist()==[12] and s.view('streak',0).tolist()==[2]
        assert s.view('free',0).tolist()==[1,3,4,0]

@pytest.mark.parametrize('fault',['negative','duplicate','range','count','dtype','strided','partition','overlap','readonly','counts_geometry','alias_command','misalignment','version'])
def test_rejects_entire_batch_before_mutating_first_layer(fault):
    s=fixture();victims,counts=packed([[0],[2]])
    if fault=='negative':victims[1,0]=-1
    elif fault=='duplicate':victims,counts=packed([[0],[2,2]])
    elif fault=='range':victims[1,0]=3
    elif fault=='count':counts[1]=2
    elif fault=='dtype':victims=victims.astype(np.int64)
    elif fault=='strided':victims=np.array([[0,0],[2,2]],np.int32)[:,::2]
    elif fault=='partition':s.owners['free'][1,0]=4
    elif fault=='overlap':s.owners['resident']=s.owners['shadow']
    elif fault=='readonly':s.owners['hot'].flags.writeable=False
    elif fault=='counts_geometry':counts=np.ones(1,np.int32)
    elif fault=='alias_command':victims=s.owners['slots'][:,:1]
    elif fault=='misalignment':victims=np.ndarray((2,1),dtype=np.int32,buffer=bytearray(9),offset=1);victims[:]=0
    else:s.owners['version'][1]=np.iinfo(np.int64).max
    before={k:a.copy() for k,a in s.owners.items()};epoch=s.epoch
    with pytest.raises((ValueError,TypeError)):s.demote(victims,counts)
    assert epoch==s.epoch and all(a.tobytes()==before[k].tobytes() for k,a in s.owners.items())

@pytest.mark.parametrize('method',['append','promote','reclaim','overflow'])
def test_phase_capacity_and_last_layer_errors_are_atomic(method):
    s=fixture();before={k:a.copy() for k,a in s.owners.items()}
    with pytest.raises(ValueError):
        if method=='append':s.append_hot(*packed([[],list(range(6))],dtype=np.int64))
        elif method=='promote':s.promote(*packed([[],[0]]))
        elif method=='reclaim':s.reclaim(*packed([[0],[2]]))
        else:
            # Exhaust the second layer's free space with legitimate cold rows.
            s.load(1,[10,20,30],list(range(7)),[False]*3,[False]*7,[False]*7,np.zeros(7,np.float32),list(range(7)),[])
            before={k:a.copy() for k,a in s.owners.items()}
            s.demote(*packed([[0],[2]]))
    assert all(a.tobytes()==before[k].tobytes() for k,a in s.owners.items())


def test_padding_and_unrelated_streak_bits_are_untouched():
    s=fixture(1);s.owners['hot'][0,3:]=-555;s.owners['grace'][0,3:]=True
    s.owners['cold'][0,4:]=-333;s.owners['streak'][0,1]=np.array([0x7fa01234],np.uint32).view(np.float32)[0]
    expected=s.owners['streak'][0,1:2].view(np.uint32).copy()
    s.demote(*packed([[0,2]]))
    assert (s.owners['hot'][0,3:]==-555).all() and s.owners['grace'][0,3:].all()
    assert (s.owners['cold'][0,4:]==-333).all()
    assert np.array_equal(s.owners['streak'][0,1:2].view(np.uint32),expected)


def test_declared_cpu_owner_bytes_and_logical_views():
    s=CpuStateArena(32,1055,1154)
    assert s.owned_bytes==32*(1055*9+1154*22+20)
    assert np.shares_memory(s.view('free',0),s.owners['free'])


def test_equal_versions_cannot_authorize_another_arenas_commands():
    a,b=fixture(),fixture()
    cmd=a.demote(*packed([[],[]]));b.demote(*packed([[],[]]))
    assert np.array_equal(a.owners['version'],b.owners['version'])
    with pytest.raises(ValueError,match='foreign'):b.validate_commands(cmd)


def test_saturated_transient_owner_reclaim_keeps_freed_slot_fifo():
    s=CpuStateArena(2,6,5)
    for l in range(2):s.load(l,[10,20,30],[100,200,300],[False]*3,[False]*3,[False]*3,np.array([1,2,3],np.float32),[4,0,2],[1,3])
    commands=s.demote(*packed([[0,2],[1]]));s.validate_commands(commands)
    assert len(s.view('cold',0))==5 and len(s.view('free',0))==0
    s.reclaim(*packed([[0,2,4],[0,1,3]]))
    assert s.view('cold',0).tolist()==[100,300,30] and s.view('slots',0).tolist()==[4,2,3]
    assert s.view('free',0).tolist()==[0,1]
    assert s.view('cold',1).tolist()==[100,200,20] and s.view('free',1).tolist()==[3,2]
    with pytest.raises(ValueError,match='Stale'):s.validate_commands(commands)


def test_live_floor_rejects_promotion_before_any_layer_changes():
    s=CpuStateArena(2,6,5)
    for l in range(2):s.load(l,[0],[1,2],[False],[False,False],[False,False],np.zeros(2,np.float32),[1,0],[2,3,4])
    before={k:a.copy() for k,a in s.owners.items()}
    with pytest.raises(ValueError,match='floor'):s.promote(*packed([[0],[0,1]]),floor=1)
    assert all(a.tobytes()==before[k].tobytes() for k,a in s.owners.items())


def test_empty_and_ragged_batch_demotion_clears_only_logical_grace():
    s=CpuStateArena(3,5,5)
    s.load(1,[0,1],[],[True,True],[],[],np.zeros(0,np.float32),[],list(range(5)))
    s.load(2,[5],[7],[True],[False],[False],np.zeros(1,np.float32),[2],[0,1,3,4])
    commands=s.demote(*packed([[],[],[0]]))
    assert commands['sizes'].tolist()==[[0,0,0],[0,0,2],[1,0,0]]
    assert s.view('grace',1).tolist()==[False,False] and s.view('cold',2).tolist()==[7,5]


def test_promote_commands_own_indices_counts_and_hot_start():
    s=CpuStateArena(1,6,5)
    s.load(0,[0],[11,12],[False],[False]*2,[False]*2,np.zeros(2,np.float32),[4,2],[0,1,3])
    rows,counts=packed([[1]])
    commands=s.promote(rows,counts)
    rows[:]=-100;counts[:]=0
    s.validate_commands(commands)
    assert commands['rows'].tolist()==[[1]] and commands['counts'].tolist()==[1]
    assert commands['hot_start'].tolist()==[1] and commands['read_slots'].tolist()==[[2]]
    assert all(not a.flags.writeable for a in commands.values() if isinstance(a,np.ndarray))
