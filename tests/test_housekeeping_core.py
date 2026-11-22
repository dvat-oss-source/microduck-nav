"""Development provenance, subset isolation, and comparison metadata."""
import json
from pathlib import Path

import pytest

from nav import bench, episode, mapsigma, provenance
from tools import fork_seeds


def save_seed(path, doc):
    doc['sha256'] = bench.seeds_digest(doc)
    path.write_text(json.dumps(doc))


def setup_preflight(tmp_path, monkeypatch):
    env = {'policy_md5': 'actual-md5', 'dynamics': 'training', 'dynamics_version': 'v'}
    monkeypatch.setattr(episode, 'read_artifacts', lambda *a: (env, dict(env)))
    monkeypatch.setattr(episode, 'artifact_digest', lambda path: 'actual-sha')
    monkeypatch.setattr(mapsigma, 'planning_lateral_sigma', lambda *a: (0.02, {
        'follower_gate_passed': True, 'file': 'follower_sim_alternate.json', 'follower_sha256': 'follower-sha'}))
    monkeypatch.setattr(provenance, 'scene_sha256', lambda *a: 'scene')
    monkeypatch.setattr(bench, '_work', lambda job: (job[1]['ep'], {
        'success': False, 'fall': False, 'hole_incursion': False, 'path_length': 1}, None))
    doc = {'nav': [{'seed': 1, 'geodesic': 1}, {'seed': 2, 'geodesic': 1}],
           'provenance': {'policy': 'primary', 'policy_md5': 'actual-md5',
                          'dynamics': 'training', 'dynamics_version': 'v', 'scene_sha256': 'scene',
                          'envelope_sha256': 'actual-sha', 'lateral_sigma': 0.02,
                          'path_smoothing_steps': 30, 'goal_dwell_s': 1.0,
                          'geodesic_method': 'footprint_16_dijkstra_los_v1'}}
    seed = tmp_path / 'seed.json'
    save_seed(seed, doc)
    class Runner:
        policy = 'primary'
        dynamics, mode, scene = 'training', 'gt', 'apartment'
        data_dir, comparison_policy = str(tmp_path), False
    return doc, seed, Runner()


def test_recorded_hashes_guard_and_older_seeds_work(tmp_path, monkeypatch):
    doc, seed, runner = setup_preflight(tmp_path, monkeypatch)
    kw = dict(seeds_path=str(seed), procs=1, reports_dir=str(tmp_path / 'out'))
    bench.run_bench(runner, 'old', **kw)
    for key in ('planner_sha256', 'mapping_sha256', 'mapsigma_sha256'):
        doc['provenance'][key] = 'wrong'
        save_seed(seed, doc)
        with pytest.raises(RuntimeError, match=key.removesuffix('_sha256')):
            bench.run_bench(runner, 'bad', **kw)
        doc['provenance'][key] = 'actual-sha'
    save_seed(seed, doc)
    bench.run_bench(runner, 'matched', **kw)


def test_subset_dev_labels_and_comparison_runtime(tmp_path, monkeypatch):
    doc, seed, runner = setup_preflight(tmp_path, monkeypatch)
    runner.policy, runner.comparison_policy = 'alternate', True
    kw = dict(seeds_path=str(seed), procs=1, reports_dir=str(tmp_path / 'out'))
    path = bench.run_bench(runner, 'subset', episodes=['nav_001'], **kw)
    report = json.loads(Path(path).read_text())
    assert 'DEV' in Path(path).name and report['DEV'] and report['dev_subset']
    assert list(report['episodes']) == ['nav_001']
    assert not report['gate']['checks']['full_frozen_suite'] and not report['gate']['passed']
    assert report['frozen_provenance'] == doc['provenance']
    assert report['comparison_runtime'] == {'policy': 'alternate', 'policy_md5': 'actual-md5',
                                            'sigma': 0.02, 'follower_sha256': 'follower-sha',
                                            'follower_gate_sha256': 'actual-sha'}
    for ids in (['nav_999'], ['nav_000', 'nav_000'], []):
        with pytest.raises(ValueError):
            bench.run_bench(runner, 'invalid', episodes=ids, **kw)
    doc['provenance']['dev'] = True
    save_seed(seed, doc)
    path = bench.run_bench(runner, 'fork', **kw)
    report = json.loads(Path(path).read_text())
    assert report['DEV'] and not report['dev_subset'] and 'DEV' in Path(path).name
    assert not report['gate']['checks']['full_frozen_suite']


def test_seed_inside_dev_directory_is_always_labeled(tmp_path, monkeypatch):
    doc, _, runner = setup_preflight(tmp_path, monkeypatch)
    monkeypatch.setattr(bench, 'ROOT', str(tmp_path))
    dev_dir = tmp_path / 'data' / 'dev'
    dev_dir.mkdir(parents=True)
    seed = dev_dir / 'untagged.json'
    save_seed(seed, doc)
    path = bench.run_bench(runner, 'dev_path', seeds_path=str(seed), procs=1,
                           reports_dir=str(tmp_path / 'out'))
    report = json.loads(Path(path).read_text())
    assert report['DEV'] and not report['dev_subset'] and 'DEV' in Path(path).name
    assert not report['gate']['checks']['full_frozen_suite']


def test_dev_fork_records_current_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(fork_seeds, 'ROOT', tmp_path)
    for name in ('planner', 'mapping', 'mapsigma', 'fsm', 'localization', 'checker', 'skills', 'episode'):
        path = tmp_path / 'nav' / (name + '.py')
        path.parent.mkdir(exist_ok=True)
        path.write_text(name)
    for name in ('envelope_primary.json', 'primitives_primary.json', 'skill_contracts.json'):
        path = tmp_path / 'data' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name)
    env = {'policy_md5': 'm', 'dynamics': 'training', 'dynamics_version': 'v', 'step_over_height': 0}
    monkeypatch.setattr(episode, 'read_artifacts', lambda *a: (env, {}))
    monkeypatch.setattr(mapsigma, 'planning_lateral_sigma', lambda *a: (0.02, {'follower_gate_passed': True}))
    monkeypatch.setattr(provenance, 'scene_sha256', lambda *a: 'scene')
    from nav import mapping, seed_validation
    monkeypatch.setattr(mapping, 'build_map', lambda *a, **k: object())
    monkeypatch.setattr(seed_validation, 'SeedValidator', lambda *a: object())
    monkeypatch.setattr(fork_seeds, 'validate_tasks', lambda *a: {'valid': True})
    p = dict(env, policy='primary', scene_sha256='scene')
    for key, path in [('envelope_sha256', 'data/envelope_primary.json'),
                      ('primitives_sha256', 'data/primitives_primary.json'),
                      ('skill_contracts_sha256', 'data/skill_contracts.json'),
                      ('checker_sha256', 'nav/checker.py')]:
        p[key] = fork_seeds.digest(tmp_path / path)
    doc = {'version': 'v3', 'master_seed': 1, 'nav': [], 'kick': [], 'pick': [], 'provenance': p}
    source = tmp_path / 'data' / 'seeds_v3.json'
    save_seed(source, doc)
    out = tmp_path / 'data' / 'dev' / 'seed.json'
    result = fork_seeds.fork(source, out, 'v4dev', 'scene', dev=True)
    assert result['provenance']['dev'] is True
    for name in ('planner', 'mapping', 'mapsigma', 'fsm', 'localization', 'skills', 'episode'):
        assert result['provenance'][name + '_sha256'] == fork_seeds.digest(tmp_path / 'nav' / (name + '.py'))
    with pytest.raises(ValueError, match='data/dev'):
        fork_seeds.fork(source, tmp_path / 'data' / 'bad.json', 'v4dev', 'scene', dev=True)
