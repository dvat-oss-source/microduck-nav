import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nav import bench, episode


def test_artifacts_precede_bounded_video_pool_and_errors_survive(tmp_path, monkeypatch):
    doc = {'nav': [{'seed': i, 'start': [0, 0, 0], 'goal': [1, 0, 0]} for i in range(3)]}
    doc['sha256'] = bench.seeds_digest(doc)
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps(doc))
    reports = tmp_path / 'reports'
    pools, closed = [], []

    class FakePool:
        def __init__(self, count):
            self.count = count
            pools.append(count)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(self.count)
        def imap_unordered(self, fn, jobs, chunksize):
            return map(fn, jobs)

    monkeypatch.setattr(bench.mp, 'get_context', lambda name: SimpleNamespace(Pool=FakePool))
    class FakeExecutor(FakePool):
        def __init__(self, max_workers, mp_context):
            super().__init__(max_workers)
            assert closed == [6]
            assert len(list((reports / 'traces' / 'run').glob('*.npz'))) == 3
            assert len(list((reports / 'failures' / 'run').glob('*.json'))) == 3
        def submit(self, fn, job):
            if job[0] == 'nav_002':
                return SimpleFuture(RuntimeError('worker process exited'))
            result = fn(job)
            return SimpleFuture(result)
    class SimpleFuture:
        def __init__(self, result): self.value = result
        def result(self):
            if isinstance(self.value, Exception): raise self.value
            return self.value
    monkeypatch.setattr(bench, 'ProcessPoolExecutor', FakeExecutor)
    monkeypatch.setattr(bench, 'as_completed', lambda futures: futures)
    def work(job):
        task = job[1]
        trace = {'qpos': np.zeros((2, 3)), 't': np.array([0., .02]), 'model_path': 'dummy'}
        return task['ep'], {'success': False, 'reasons': ['test'], 'fallen': False,
                            'hole_incursion': False, 'path_length': 0., 'time_to_goal': None}, trace
    monkeypatch.setattr(bench, '_work', work)
    seen = []
    def render(task, trace, path):
        seen.append((task['ep'], trace['qpos'].shape, trace['model_path']))
        if task['ep'] == 'nav_001':
            raise RuntimeError('render failure')
        Path(path).write_bytes(b'video')
    monkeypatch.setattr(episode, 'failure_video', render)
    output = bench.run_bench(lambda task: None, 'run', seeds_path=str(seeds), procs=6,
                             reports_dir=str(reports), video_hook=render)
    assert pools == [6, 3] and closed == [6, 3]
    assert seen == [('nav_000', (2, 3), 'dummy'), ('nav_001', (2, 3), 'dummy')]
    report = json.loads(Path(output).read_text())
    assert report['episodes']['nav_000']['failure_video'].endswith('nav_000.mp4')
    assert 'render failure' in report['episodes']['nav_001']['video_error']
    assert 'worker process exited' in report['episodes']['nav_002']['video_error']
    failure = json.loads((reports / 'failures' / 'run' / 'nav_001.json').read_text())
    assert 'render failure' in failure['verdict']['video_error']


def test_unpickleable_custom_hook_stays_serial_with_multiple_episode_workers(tmp_path, monkeypatch):
    doc = {'nav': [{'seed': 0, 'start': [0, 0, 0], 'goal': [1, 0, 0]}]}
    doc['sha256'] = bench.seeds_digest(doc)
    seeds = tmp_path / 'seeds.json'
    seeds.write_text(json.dumps(doc))
    class FakePool:
        def __init__(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def imap_unordered(self, fn, jobs, chunksize): return map(fn, jobs)
    monkeypatch.setattr(bench.mp, 'get_context', lambda name: SimpleNamespace(Pool=FakePool))
    monkeypatch.setattr(bench, '_work', lambda job: (job[1]['ep'],
        {'success': False, 'fallen': False, 'hole_incursion': False, 'path_length': 0., 'time_to_goal': None},
        {'qpos': np.zeros((2, 3)), 't': np.array([0., .02]), 'model_path': 'dummy'}))
    def no_executor(*args, **kwargs):
        raise AssertionError('custom hook must not use renderer workers')
    monkeypatch.setattr(bench, 'ProcessPoolExecutor', no_executor)
    seen = []
    hook = lambda task, trace, path: seen.append(task['ep'])
    output = bench.run_bench(lambda task: None, 'serial', seeds_path=str(seeds), procs=2,
                             reports_dir=str(tmp_path), video_hook=hook)
    assert seen == ['nav_000']
    assert json.loads(Path(output).read_text())['episodes']['nav_000']['failure_video'].endswith('nav_000.mp4')
