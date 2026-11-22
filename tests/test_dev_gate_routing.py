"""Development gates remain guarded and cannot silently certify frozen runs."""
import json
from pathlib import Path

import pytest

from nav import bench, mapsigma
from tools import fork_seeds
from tools import planner_gate
from test_housekeeping_core import setup_preflight


def test_planner_gate_uses_same_dev_directory_for_report_and_sigma(tmp_path, monkeypatch):
    """The gate must certify the candidate's uncertainty, not the frozen follower's."""
    monkeypatch.setattr(planner_gate, 'ROOT', tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    env = {'policy_md5': 'measured-policy', 'dynamics': 'training',
           'dynamics_version': 'version', 'step_over_height': 0.0}
    (data / 'envelope_test.json').write_text('{}')
    (data / 'primitives_test.json').write_text('{}')
    dev = tmp_path / 'dev_gates'
    dev.mkdir()
    gate = {**env, 'passed': True}
    (dev / 'follower_sim_test.json').write_text(json.dumps(gate))
    monkeypatch.setattr(planner_gate, 'read_artifacts', lambda policy: (env, {}))
    calls = []

    def sigma(policy, measured, reports_dir):
        calls.append((policy, measured, reports_dir))
        return 0.012, {'follower_gate_passed': True}

    monkeypatch.setattr(planner_gate, 'planning_lateral_sigma', sigma)
    sentinel = object()
    map_inputs = []

    def build_map(**kwargs):
        map_inputs.append(kwargs)
        return sentinel

    monkeypatch.setattr(planner_gate, 'build_map', build_map)
    _, _, _, grid, uncertainty, provenance = planner_gate.load_setup('test', dev)
    assert grid is sentinel and uncertainty == 0.012
    assert calls == [('test', env, dev)]
    assert map_inputs == [{'lateral_sigma': 0.012, 'step_over_height': 0.0}]
    assert provenance['follower_gate_passed']
    assert not (tmp_path / 'reports').exists()


def test_planner_gate_dev_input_requires_separate_output_directory():
    with pytest.raises(SystemExit) as error:
        planner_gate.main(['--follower-reports-dir', 'candidate-gates'])
    assert error.value.code == 2


@pytest.mark.parametrize('use_alias', [False, True])
def test_planner_gate_dev_input_cannot_overwrite_canonical_reports(tmp_path, monkeypatch, use_alias):
    monkeypatch.setattr(planner_gate, 'ROOT', tmp_path)
    canonical = tmp_path / 'reports'
    canonical.mkdir()
    output = canonical
    if use_alias:
        output = tmp_path / 'reports_alias'
        output.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(SystemExit) as error:
        planner_gate.main(['--follower-reports-dir', str(tmp_path / 'dev'),
                           '--reports-dir', str(output)])
    assert error.value.code == 2


def test_custom_gate_directory_is_explicit_and_dev_only(tmp_path, monkeypatch):
    _, seed, runner = setup_preflight(tmp_path, monkeypatch)
    runner.follower_reports_dir = str(tmp_path / 'dev_gates')
    kw = dict(seeds_path=str(seed), procs=1, reports_dir=str(tmp_path / 'out'))
    with pytest.raises(ValueError, match='development only'):
        bench.run_bench(runner, 'frozen', **kw)
    calls = []
    def current_gate(policy, env, reports_dir):
        calls.append(reports_dir)
        return .02, {'follower_gate_passed': True}
    monkeypatch.setattr(mapsigma, 'planning_lateral_sigma', current_gate)
    report = json.loads(Path(bench.run_bench(runner, 'dev', episodes=['nav_000'], **kw)).read_text())
    assert calls == [runner.follower_reports_dir]
    assert report['DEV'] and not report['gate']['checks']['full_frozen_suite']


def test_nondev_fork_cannot_select_alternate_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_seeds, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='require --dev'):
        fork_seeds.fork(tmp_path / 'missing.json', tmp_path / 'out.json', 'v4', 'scene',
                        follower_reports_dir=tmp_path / 'gates')
