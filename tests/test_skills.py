import json
import numpy as np
from nav.skills import (CONTRACTS_PATH, KICK_SPECS, kick_pre_pose, kick_precondition,
                        run_kick_trial, run_recover_trial)
from nav.sim import Sim


def test_contracts_report_measurement_limits():
    c = json.loads(CONTRACTS_PATH.read_text())
    assert not c['fallback']
    assert c['kick']['left']['mirror']
    assert not c['pick']['measured']['object_lift_tested']
    assert len(c['recover']['measured']['trials']) >= 4
    for side in ('left', 'right'):
        pose = kick_pre_pose((1., 2.), .3, side, c)
        assert kick_precondition(pose, (1., 2.), side, c)[0]


def test_mirrored_kick_and_fallen_recovery_run():
    s = Sim('flat', ball=True)
    r = run_kick_trial(s, KICK_SPECS['left_mirror_pollen'], seed=0)
    assert r['handoff_ok'] and r['success'] and not r['fell']
    recovery = run_recover_trial(s, 'atlantis_standup_flat', 'back', 0)
    assert recovery['start_fallen']
    assert isinstance(recovery['success'], bool)
