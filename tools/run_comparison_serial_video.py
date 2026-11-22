"""Frozen nav comparison with serial videos, safe beside a two-process reference replay.

The wrapper deliberately uses run_bench's documented serial custom-hook path. It
calls the identical native failure_video function with unchanged frames/JSON.
Physics: four workers + parent + tracker = six; video: parent + one encoder.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.bench import run_bench
from nav.episode import EpisodeRunner, failure_video


def serial_video(trace, task, out):
    return failure_video(trace, task, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    runner = EpisodeRunner(policy=args.policy, mode="gt", dynamics="training", comparison_policy=True)
    print(run_bench(runner, args.run, seeds_path=args.seeds, procs=4,
                    video_hook=serial_video, task_types=["nav"]), flush=True)


if __name__ == "__main__":
    main()
