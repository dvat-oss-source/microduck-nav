"""Measurement contracts: reject immobile policies, preserve measured frames."""
import math
import numpy as np
import pytest
from nav.measure import accepted_walkers, command_progress, primitive_commands, rollout, select_walker, wilson_upper
from nav.sim import Sim, TRAINING_DYNAMICS_VERSION


def test_progress_cannot_promote_a_stationary_checkpoint():
    assert not command_progress([.3,0,0],[0,0,0])
    assert not command_progress([.3,0,.5],[.2,0,-.1])
    assert command_progress([.3,0,.5],[.1,0,.3])
    lock={'good':{'kind':'walk'},'bad':{'kind':'walk','status':'rejected'},'skill':{'kind':'pick'}}
    assert accepted_walkers(lock)==['good']


def test_zero_is_not_substituted_for_a_motion_primitive():
    assert primitive_commands({'safe_cmds':[[0,0,0]]})==[]
    commands=primitive_commands({'safe_cmds':[[0,0,0],[.2,0,0],[0,0,-.5]]})
    assert commands==[[.2,0,0],[0,0,-.5]]


def test_heading_wrap_and_world_aligned_endpoint():
    s=Sim('flat','walk',dynamics='training',record=False)
    p,_,_=rollout(s,0,[([.3,0,0],.2)],heading=1.5*math.pi)
    assert p[0,2]==0
    assert abs(p[-1,2])<math.pi
    assert p[-1,1]<0


def test_gate_and_small_sample_uncertainty():
    assert abs(wilson_upper(0,8)-.32440756)<1e-6
    base={'dynamics':'training','dynamics_version':TRAINING_DYNAMICS_VERSION,'grid_fall_rate':0,'safe_commands':10,'max_forward_speed':.2,'usable_primitives':10}
    rows=[dict(base,key='stationary',forward_safe_commands=0),dict(base,key='moving',forward_safe_commands=2)]
    assert select_walker(rows)['selected']=='moving'
    assert select_walker(rows[:1])['gate']=='FAIL'


def test_gate_refuses_missing_or_mixed_dynamics_versions():
    row = {'key':'test', 'dynamics':'training', 'forward_safe_commands':1,
           'usable_primitives':1, 'grid_fall_rate':0., 'safe_commands':1, 'max_forward_speed':.1}
    with pytest.raises((ValueError, RuntimeError)):
        select_walker([row])
    with pytest.raises((ValueError, RuntimeError)):
        select_walker([dict(row, dynamics_version=TRAINING_DYNAMICS_VERSION),
                       dict(row, dynamics_version='stale')])
