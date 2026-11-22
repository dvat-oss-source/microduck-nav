"""Run the connected stack on frozen tasks, locally with at most nine CPUs."""
import argparse
import datetime
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.bench import run_bench
from nav.episode import ABLATIONS, EpisodeRunner, failure_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", choices=["gt", "odom", "odom+fiducial", "learned+fiducial"], default="gt")
    ap.add_argument("--ablation", choices=sorted(ABLATIONS), default="legacy",
                    help="Stage 1 ablation: 1a=learned+fiducial; 1a+1c=+stop-and-look; all=+anchors; "
                         "1b+1c=odom+fiducial with look and anchors. legacy=live behaviour (default)")
    ap.add_argument("--dynamics", choices=["infer", "training"], default="training")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-time", type=float, default=120.0)
    ap.add_argument("--seeds", default="data/seeds_v1.json")
    ap.add_argument("--run")
    ap.add_argument("--tasks", nargs="+", choices=["nav", "kick", "pick"])
    ap.add_argument("--episodes", nargs="+", help="development subset of explicit episode IDs")
    ap.add_argument("--reports-dir", default="reports", help="output directory (use reports/dev/<fix> for development)")
    ap.add_argument("--follower-reports-dir", help="development: directory with current passing follower gates")
    ap.add_argument("--comparison-policy", action="store_true",
                    help="same frozen nav seeds, another verified walker and its own passing follower map")
    args = ap.parse_args()
    runner = EpisodeRunner(policy=args.policy, mode=args.mode, dynamics=args.dynamics,
                           max_time=args.max_time, comparison_policy=args.comparison_policy,
                           follower_reports_dir=args.follower_reports_dir, ablation=args.ablation)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "" if args.ablation == "legacy" else f"_abl{args.ablation.replace('+', '_')}"
    run = args.run or f"{args.policy}_{args.mode.replace('+', '_')}{tag}_{args.dynamics}_{stamp}"
    output = run_bench(runner, run, seeds_path=args.seeds, procs=args.procs, limit=args.limit,
                       video_hook=failure_video, task_types=args.tasks, episodes=args.episodes,
                       reports_dir=args.reports_dir)
    print(output)
    print(json.dumps(json.loads(Path(output).read_text())["by_task"], indent=2))


if __name__ == "__main__":
    main()
