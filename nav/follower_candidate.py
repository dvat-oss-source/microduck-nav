"""Experimental apartment rotation-progress monitor; never a certified benchmark follower.

The canonical nav.follower implementation and its hash remain untouched. This
subclass is injected explicitly by a development-only driver. Escalated pure-yaw
commands are exact entries in the verified envelope's safe_cmds list.
"""
import numpy as np

from .follower import RPPFollower


class RPPFollowerCandidate(RPPFollower):
    def set_path(self, poses):
        super().set_path(poses)
        self._rotation_floor = {1: 0.0, -1: 0.0}
        self._rotation_attempt = None
        self.escalation_events = []
        safe = self.env.safe
        self._rotation_levels = {1: [], -1: []}
        if safe is not None and self.model.measured:
            for sign in (1, -1):
                self._rotation_levels[sign] = sorted({float(cmd[2]) for cmd in safe
                    if abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9 and sign * cmd[2] > 1e-9},
                    key=abs)
        # Wait two measured response horizons before judging a rotation command.
        self.rotation_window = 2.0 * max(self.dt, float(self.env.delay[0] + self.env.tau[0]),
                                          float(self.env.delay[2] + self.env.tau[2]))

    def _rotate_cmd(self, err):
        command = super()._rotate_cmd(err)
        sign = 1 if err > 0 else -1
        floor = self._rotation_floor[sign]
        if abs(floor) > abs(command[2]):
            command[2] = floor
        return command

    def compute(self, pose2_est, gt_vel_or_None=None):
        command, status = super().compute(pose2_est, gt_vel_or_None)
        pure_yaw = (abs(command[0]) < 1e-9 and abs(command[1]) < 1e-9 and abs(command[2]) > 1e-9)
        if not pure_yaw or not self.hist:
            self._rotation_attempt = None
            return command, status
        sign = 1 if command[2] > 0 else -1
        now, _, _, heading = self.hist[-1]
        previous = self._rotation_attempt
        if previous is None or previous['sign'] != sign or abs(previous['command'] - command[2]) > 1e-9:
            self._rotation_attempt = {'sign': sign, 't': now, 'heading': heading, 'command': float(command[2])}
            return command, status
        elapsed = now - previous['t']
        if elapsed < self.rotation_window:
            return command, status
        achieved = sign * (heading - previous['heading'])
        predicted_rate = abs(float(self.model.forward(0.0, command[2])[1]))
        expected = predicted_rate * elapsed
        # A quarter of the measured angular response is a generic progress floor.
        if achieved < 0.25 * expected:
            stronger = [level for level in self._rotation_levels[sign] if abs(level) > abs(command[2]) + 1e-9]
            if stronger:
                level = stronger[0]
                self._rotation_floor[sign] = level
                self.escalation_events.append({'t': now, 'state': self.state, 'from_wz': float(command[2]),
                    'to_wz': level, 'elapsed_s': elapsed, 'observed_yaw_rad': achieved,
                    'predicted_yaw_rad': expected, 'window_s': self.rotation_window})
                command = np.array([0.0, 0.0, level])
                # Do not call _out twice: the base computation has already advanced
                # one controller tick. Keep its predictor's command history accurate.
                self.last_cmd = command.copy()
        self._rotation_attempt = {'sign': sign, 't': now, 'heading': heading, 'command': float(command[2])}
        return command, status
