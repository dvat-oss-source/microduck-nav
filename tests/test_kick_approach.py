from types import SimpleNamespace
import numpy as np
from nav.mapping import GridMap, overlay_discs
from nav.skills import kick_stage_plan, kick_pre_pose, KickApproach, load_contracts


def grid():
    shape = (200,200)
    z = np.zeros(shape,bool)
    return GridMap(.02,(-2.,-2.),z.copy(),z.copy(),z.copy(),np.zeros(shape,np.float32),
                   np.ones(shape,np.float32),params=dict(r_inflate=.16,decay_k=5.))


def test_full_object_inflation_is_private_and_monotone():
    gm = grid()
    overlay = overlay_discs(gm,[dict(name='ball',xy=[0.,0.],radius=.04),
                               dict(name='box',xy=[1.,0.],radius=.06)])
    assert overlay.is_lethal(0.,0.) and overlay.is_lethal(.19,0.) and overlay.is_lethal(1.2,0.)
    assert not overlay.is_lethal(.3,0.)
    assert not gm.occ.any() and not gm.lethal.any() and not gm.cost.any()
    assert overlay.params['r_inflate'] == gm.params['r_inflate']


def test_stage_is_behind_and_outside_full_ball_inflation():
    gm = grid()
    info = kick_stage_plan(gm,(-1.,-.5,0.),(0.,0.),0.)
    assert info['ok'] and info['backoff'] > 0
    assert info['stage_pose'][0] <= -.25
    assert not info['map'].is_lethal(*info['stage_pose'][:2])
    assert info['map'].is_lethal(*info['pre_pose'][:2])
    assert info['runin_map'] is gm
    assert not kick_stage_plan(gm,(.1,0.,0.),(0.,0.),0.)['ok']


def test_stage_respects_other_objects_in_runin():
    gm = overlay_discs(grid(),[dict(xy=[-.2,0.],radius=.06)])
    assert not kick_stage_plan(gm,(-1.,-.5,0.),(0.,0.),0.)['ok']


def approach():
    contracts = load_contracts()
    follower = SimpleNamespace(env=SimpleNamespace(safe=np.array([[.1,0.,0.],[-.1,0.,0.],[0.,0.,0.]])),
        model=SimpleNamespace(v_at=lambda vx:vx*.2,stop_per_v=.3,vx_min=.1,vx_start=.1,w_turn_max=1.,
            inverse=lambda v,w,vx_lo:(vx_lo,w,v,w)))
    ap = KickApproach(contracts,'right',0.,follower)
    pre = kick_pre_pose((0.,0.),0.,'right',contracts)
    stage = (pre[0]-.25,pre[1],pre[2])
    ap.begin(stage,(0.,0.),0.)
    return ap,pre,stage


def test_runin_is_straight_and_hands_off_on_current_box_without_dwell():
    ap,pre,stage = approach()
    command,status = ap.step(stage,(0.,0.),.02)
    assert status == 'advance' and np.array_equal(command,[.1,0.,0.])
    command,status = ap.step(pre,(0.,0.),.04)
    assert status == 'ready' and np.array_equal(command,np.zeros(3))
    assert ap.events[-1]['event'] == 'ready'


def test_ball_displacement_aborts_and_overshoot_never_turns_in_place():
    ap,pre,stage = approach()
    command,status = ap.step(stage,(.021,0.),.02)
    assert status == 'retarget' and not command.any()
    ap,pre,stage = approach()
    command,status = ap.step((pre[0]+.1,pre[1],pre[2]),(0.,0.),.02)
    assert status == 'retreat' and command[0] < 0 and command[1] == command[2] == 0


def test_runin_corrects_cross_track_error_with_positive_forward_motion():
    ap,pre,stage = approach()
    command,status = ap.step((stage[0],stage[1]+.015,stage[2]),(0.,0.),.02)
    assert status == 'advance' and command[0] > 0 and command[2] < 0
