"""Summarize fixed flat probes without treating them as apartment replay."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'reports/fid_schedule.json').read_text())
ref=json.loads((ROOT/'reports/fid_reference_nominal.json').read_text())
r=np.asarray(ref['log'],float)
windows=[(50,201),(350,501)]; idx=np.r_[50:201,350:501]
report={'protocol':{'reference':'unmodified cb8561f6 mjlab task, Warp CPU, nominal flat tiles',
 'sim_scene':'flat','num_schedules':10,'replicates':2,'seconds':10,'seed':0,
 'steady_windows_s':[[1,4],[7,10]],'commands_start_at_s':[0,6],'stop_start_at_s':4,
 'reference_wall_s':ref['wall_s'],'limitations':['No complete training DR/corruption.','Delay RNG is independent between implementations.','Reset BAM target/cache initialization is not identical: mjlab writes zero absolute targets once, standalone starts with DEFAULT_POSE.','Only two repetitions per command, stochastic gait initiation.','Generated reference flat tiles differ from nav plane.','Not a same-scene apartment outcome replay.']},
 'modes':{},'gate':'PARTIAL_AGREEMENT_NOT_CERTIFIED','independent_apartment_replay_gate':'NOT_RUN'}
for mode in ('infer','training'):
 local=json.loads((ROOT/f'reports/fid_nav_{mode}.json').read_text());a=np.asarray(local['log'],float)
 rd=r[-1,:,:2]-r[0,:,:2];ad=a[-1,:,:2]-a[0,:,:2];xy=np.linalg.norm(rd-ad,axis=1)
 ry=np.unwrap(r[:,:,2],axis=0);ay=np.unwrap(a[:,:,2],axis=0);yaw=np.abs((ay[-1]-ay[0])-(ry[-1]-ry[0]))
 rows=[]
 for j in range(10):
  ids=[j,j+10];rv=r[idx][:,ids,3:].mean((0,1));av=a[idx][:,ids,3:].mean((0,1));c=spec['envs'][j]['schedule'][0][1:]
  active=[k for k in range(3) if abs(c[k])>1e-9 and abs(rv[k])>.03]
  relative={('vx','vy','wz')[k]:float(abs(av[k]-rv[k])/abs(rv[k])) for k in active}
  rows.append({'name':spec['envs'][j]['name'],'command':c,'reference_velocity':rv.tolist(),'sim_velocity':av.tolist(),
               'abs_velocity_error':np.abs(av-rv).tolist(),'relative_error_active_moving_axes':relative})
 report['modes'][mode]={'falls':sum(x is not None for x in local['fall_t']),
  'reference_falls':sum(x is not None for x in ref['fall_t']),
  'endpoint_xy_error_m':{'median':float(np.median(xy)),'max':float(max(xy))},
  'endpoint_yaw_error_rad':{'median':float(np.median(yaw)),'max':float(max(yaw))},
  'pooled_command_velocity':rows,'wall_s':local['wall_s']}
(ROOT/'reports/sim_fidelity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=['# Simulation fidelity: explicit nominal training mode','',
 'Default `Sim(..., dynamics="infer")` preserves deployment rehearsal parity. `dynamics="training"` adds the cb8561f6 backlash robot, full collision configuration, training solver, 1.75 A current limiter, encoder-through-backlash feedback, 3–6 physics-substep actuator delay, 0–1 control-step gyro/gravity delays, and one-control-step joint-speed delay.',
 '', 'The training mode is nominal structure with seeded voltage/sag, not complete mass/CoM/friction/armature/IMU/encoder/noise domain randomization. Robot source/config targets Pollen alpha; current user checkpoint training commits were not independently recovered. Reset BAM target/cache initialization is not identical: mjlab writes zero absolute targets once; standalone starts with DEFAULT_POSE. This may affect gait initiation (inference, not isolated experimentally).',
 '', 'Reference: tools/fid_mjlab_ref.py, 20 independent mjlab envs × 10 s on Warp CPU (53.72 s), 10 fixed schedules repeated twice. Same command timing, nominal 7.4 V / 0.1 sag, no pushes or observation corruption. Means pool windows 1–4 s and 7–10 s.',
 '', '| command | mjlab vx / wz | training vx / wz | infer vx / wz |','|---|---|---|---|']
for j in (2,6,8,9):
 vals=[report['modes'][m]['pooled_command_velocity'][j] for m in ('training','infer')];rv=vals[0]['reference_velocity'];a=vals[0]['sim_velocity'];b=vals[1]['sim_velocity']
 lines.append(f"| {vals[0]['command']} | {rv[0]:.3f} / {rv[2]:.3f} | {a[0]:.3f} / {a[2]:.3f} | {b[0]:.3f} / {b[2]:.3f} |")
lines += ['', 'All 20 probes are fall-free in both implementations. Pooled moving commanded-axis response agrees within 10%; this excludes near-zero reference responses and does not establish trajectory parity. Positive yaw restarts show stochastic initiation: reference last-window speeds 0.798/0.783 vs training 0.793/0.006 rad/s; first-window reference 0.845/0.005 vs training 0.777/0.768. Forward straight yaw drift also differs (mjlab −0.074 vs training +0.044 rad/s).',
 '', 'Training endpoint XY error: median 9.6 mm, max 0.534 m; yaw error median 0.035 rad, max 0.936 rad. Infer max XY/yaw errors are 0.857 m / 5.43 rad. The unmodified infer deployment loop often fails to initiate the reference gait for arc commands. These tests justify measuring the training structural mode separately while preserving the original loop.',
 '', '**Gate: partial agreement, not certified.** Two repetitions per command cannot certify initiation probabilities or rare falls. No independent same-scene apartment outcome replay was performed here; H remains untrusted until that separate 20-episode replay requirement is met.',
 '', 'Reproduce: from ~/microduck, `OMP_NUM_THREADS=1 .venv-ref/bin/python nav/tools/fid_mjlab_ref.py nav/reports/fid_schedule.json nominal nav/reports/fid_reference_nominal.json`; from nav, `PYTHONPATH=. $PY tools/fid_nav.py reports/fid_schedule.json training reports/fid_nav_training.json` (and infer), then `$PY tools/fid_compare.py`.']
(ROOT/'reports/sim_fidelity.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({m:{k:v for k,v in d.items() if k!='pooled_command_velocity'} for m,d in report['modes'].items()},indent=2))
