"""Controlled flat-scene steering calibration, independent of benchmark episodes.

Only exact envelope-safe straight/forward-turn commands are applied. This is a
CPU measurement, not training; existing envelope/primitive artifacts are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nav.episode import read_artifacts
from nav.sim import Sim, CONTROL_DT

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('atlantis_walk', 'walk', 'velocity_hard')
SEEDS = (1001, 1002, 1003)


def yaw_oscillation(t, yaw):
    """Steady yaw residual after removing its fitted mean turn rate."""
    t, yaw = np.asarray(t), np.unwrap(yaw)
    fit = np.polyfit(t - t[0], yaw, 1)
    residual = yaw - np.polyval(fit, t - t[0])
    frequencies = np.fft.rfftfreq(len(residual), CONTROL_DT)
    # The derivative spectrum finds the periodic stepping component rather than
    # interpreting slow yaw wander as one extremely long gait cycle. The angular
    # residual bound below still includes that wander, conservatively.
    spectrum = np.abs(np.fft.rfft(residual)) ** 2 * frequencies ** 2
    peak = 1 + int(np.argmax(spectrum[1:]))
    return {'rms_rad': float(np.sqrt(np.mean(residual ** 2))),
            'abs_p95_rad': float(np.percentile(np.abs(residual), 95)),
            'gait_period_s': float(1 / frequencies[peak]),
            'mean_yaw_rate_rad_s': float(fit[0])}


def step_response(previous, current, gait_period):
    """Causal gait-period averaged rate, 63.2% response with one-period persistence.

    The reported time includes the causal averaging delay, conservatively. A step
    that never reaches/persists above the threshold is reported as unmeasured.
    """
    old = np.asarray(previous)
    cur = np.asarray(current)
    width = max(1, int(round(gait_period / CONTROL_DT)))
    previous_rate = float(np.median(old[-max(width, len(old) // 4):, 2]))
    steady_rate = float(np.median(cur[-max(width, len(cur) // 4):, 2]))
    change = steady_rate - previous_rate
    if abs(change) <= 1e-9:
        return None
    smooth = np.convolve(cur[:, 2], np.ones(width) / width, mode='valid')
    fraction = (smooth - previous_rate) / change
    target = 1. - np.exp(-1.)
    for i in range(max(0, len(fraction) - width + 1)):
        if np.all(fraction[i:i + width] >= target):
            return {'response_s': float((i + width) * CONTROL_DT),
                    'baseline_rate_rad_s': previous_rate, 'steady_rate_rad_s': steady_rate,
                    'averaging_window_s': width * CONTROL_DT}
    return None


def measure(policy):
    envelope, primitives = read_artifacts(policy)
    seconds = float(envelope['measurement']['seconds_per_command'])
    # The forward-turn primitive experiment already used these command levels.
    heading0 = [p for p in primitives['primitives'] if p['start_heading_idx'] == 0]
    turn_commands = {tuple(p['cmd_sequence'][0][:3]) for p in heading0
                     if len(p['cmd_sequence']) == 1 and p['cmd_sequence'][0][0] > 0
                     and abs(p['cmd_sequence'][0][1]) < 1e-9
                     and abs(p['cmd_sequence'][0][2]) > 0}
    positive = min((c for c in turn_commands if c[2] > 0), key=lambda c: abs(c[2]))
    negative = min((c for c in turn_commands if c[2] < 0), key=lambda c: abs(c[2]))
    straight = (positive[0], 0., 0.)
    commands = [straight, positive, straight, negative, straight]
    safe = {tuple(c) for c in envelope['safe_cmds']}
    if any(c not in safe for c in commands):
        raise RuntimeError('controlled experiment contains a non-safe command')
    sim = Sim('flat', policy, record=False, dynamics='training')
    trials, noise, responses = [], [], []
    for seed in SEEDS:
        gt = sim.reset(seed, start_pose=(0., 0., 0.))
        stages = []
        for command in commands:
            samples = []
            for _ in range(int(round(seconds / CONTROL_DT))):
                before = float(gt.pose2[2])
                gt, _ = sim.step(np.asarray(command))
                rate = (float(gt.pose2[2]) - before + np.pi) % (2 * np.pi) - np.pi
                samples.append([float(gt.t), float(gt.pose2[2]), rate / CONTROL_DT])
                if gt.fallen:
                    raise RuntimeError(f'{policy} seed={seed} fell during calibration')
            # The second half is steady measurement; no response transient is used.
            tail = np.asarray(samples)[len(samples) // 2:]
            oscillation = yaw_oscillation(tail[:, 0], tail[:, 1])
            noise.append(oscillation)
            stages.append({'command': list(command), 'samples_t_yaw_rate': samples,
                           'steady_yaw_oscillation': oscillation})
        for before, after in zip(stages, stages[1:]):
            period = max(before['steady_yaw_oscillation']['gait_period_s'],
                         after['steady_yaw_oscillation']['gait_period_s'])
            response = step_response(before['samples_t_yaw_rate'], after['samples_t_yaw_rate'], period)
            after['step_response'] = response
            if response is not None:
                responses.append(response['response_s'])
        trials.append({'seed': seed, 'stages': stages})
        print(f'{policy} seed={seed} measured {len(commands)*seconds:.1f}s', flush=True)
    if len(responses) != len(SEEDS) * (len(commands) - 1):
        raise RuntimeError(f'{policy}: not all controlled yaw steps have a measurable response')
    # Quantiles are declared before any controller/dev evaluation. Use conservative
    # 95th percentiles of directly observed noise and response, without seed tuning.
    calibration = {'heading_deadband_rad': float(np.percentile([n['abs_p95_rad'] for n in noise], 95)),
                   'yaw_response_s': float(np.percentile(responses, 95)),
                   'gait_period_s': float(np.median([n['gait_period_s'] for n in noise])),
                   'forward_response_s': float(envelope['fol']['vx']['delay'] + envelope['fol']['vx']['tau'])}
    return {'policy': policy, 'policy_md5': envelope['policy_md5'], 'scene': 'flat',
            'dynamics': 'training', 'dynamics_version': envelope['dynamics_version'],
            'seconds_per_stage': seconds, 'experimental_seeds': list(SEEDS),
            'envelope_sha256': hashlib.sha256((ROOT / f'data/envelope_{policy}.json').read_bytes()).hexdigest(),
            'primitives_sha256': hashlib.sha256((ROOT / f'data/primitives_{policy}.json').read_bytes()).hexdigest(),
            'calibration': calibration, 'trials': trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/dev/A5/steering_response_measurements.json')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error(f'refusing to overwrite measurement evidence: {path}')
    result = {'dev': True, 'benchmark_seeds_used': False,
              'method': 'Exact safe forward/turn primitive command levels; 4s envelope measurement windows; '
                        'steady half-window detrended yaw residual; causal gait-period averaged63.2% step '
                        'response persisting one gait period;95th-percentile noise and response bounds.',
              'policies': {p: measure(p) for p in POLICIES}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
