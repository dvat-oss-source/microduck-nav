"""Reproducible CPU command-envelope and motion-primitive measurements.

Run ``python -m nav.measure --workers 3 --dynamics training``.  Every accepted
walk policy is measured independently; rejected checkpoints are never selected.
``training`` is nominal cb8561f6 structure, delays and battery variation, not the
complete randomized training distribution. Confidence bounds describe these
flat-ground probes only. No claim of a real-robot safe envelope is made.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import math
from pathlib import Path
import time
import numpy as np
from . import policies
from .sim import Sim, CONTROL_DT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYS = ('walk', 'velstand_flat', 'velstand_rough', 'velocity_hard')
GRID = tuple(product((-.3, 0., .1, .2, .3, .4, .5), (-.2, -.1, 0., .1, .2), (-1.5, -1., -.5, 0., .5, 1., 1.5)))

def wilson_upper(falls, n):
    if not n:
        return 1.
    z=1.959963984540054; p=falls/n; d=1+z*z/n
    return float((p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d)

def accepted_walkers(lock=None):
    return [k for k,v in (policies.load_lock() if lock is None else lock).items()
            if not k.startswith('_') and v.get('kind') == 'walk' and v.get('status') != 'rejected']

def rollout(sim, seed, sequence, heading=0., extra=None):
    gt=sim.reset(seed, (0.,0.,heading), extra=extra)
    poses=[gt.pose2.copy()]; velocities=[]; falls=False
    for cmd, seconds in sequence:
        for _ in range(round(seconds/CONTROL_DT)):
            gt,_=sim.step(cmd)
            poses.append(gt.pose2.copy())
            velocities.append([*gt.lin_vel_body[:2], gt.ang_vel_body[2]])
            if gt.fallen or not np.all(np.isfinite(gt.pose2)):
                falls=True
                break
        if falls:
            break
    p=np.asarray(poses); p[:,2]=np.unwrap(p[:,2]); p[:,2]-=p[0,2]
    return p, np.asarray(velocities), falls

def command_progress(cmd, velocity):
    """Require useful signed response on every commanded component."""
    c=np.asarray(cmd); v=np.asarray(velocity)
    active=np.abs(c)>1e-9
    return bool(np.all(v[active]*np.sign(c[active]) >= np.maximum(.05*np.abs(c[active]), np.array([.005,.005,.03])[active])))

def fit_fol(traces, command, step_start=.5):
    from scipy.optimize import least_squares
    out={}
    for a,name in enumerate(('vx','vy','wz')):
        if not traces[a]:
            out[name]={'gain':0.,'tau':0.,'delay':0.,'fit_rmse':None}; continue
        y=np.mean(traces[a],axis=0)[:,a]
        # Average over a 0.2 s gait window; fit the response, not gait sway.
        kernel=np.ones(10)/10
        y=np.convolve(np.pad(y,(5,4),mode='edge'),kernel,mode='valid')
        t=np.arange(len(y))*CONTROL_DT-step_start
        def predict(x):
            gain,tau,delay,bias=x
            return bias+gain*command[a]*(1-np.exp(-np.maximum(t-delay,0)/tau))
        res=least_squares(lambda x: predict(x)-y,[.7,.2,.04,0],bounds=([0,.02,0,-.1],[4,2,.8,.1]))
        gain,tau,delay,bias=res.x
        out[name]={'gain':float(gain),'tau':float(tau),'delay':float(delay),
                   'bias':float(bias),'fit_rmse':float(np.sqrt(np.mean(res.fun**2)))}
    return out

def measure_policy(key, dynamics='training', seeds=8, seconds=4., commands=GRID, primitive_seeds=8):
    start=time.time(); sim=Sim('flat',key,dynamics=dynamics,record=False)
    rows=[]; safe=[]
    for ci,cmd in enumerate(commands):
        ends=[]; vels=[]; falls=0
        for seed in range(seeds):
            p,v,f=rollout(sim,seed,[(cmd,seconds)])
            falls+=f; ends.append(p[-1]); vels.append(v[len(v)//2:].mean(0))
        vm=np.mean(vels,0); useful=command_progress(cmd,vm)
        zero=bool(np.max(np.abs(cmd))==0)
        good=falls==0 and (useful or zero)
        if good:
            safe.append(list(cmd))
        rows.append({'command':list(cmd),'falls':falls,'trials':seeds,'fall_rate':falls/seeds,
                     'fall_rate_wilson95_upper':wilson_upper(falls,seeds),'mean_velocity':vm.tolist(),
                     'end_mean':np.mean(ends,0).tolist(),'end_std':np.std(ends,0).tolist(),
                     'progress':useful,'safe_observed':good})
        if ci%40==0:
            print(key,'grid',ci,'/',len(commands),round(time.time()-start,1),flush=True)
    safe_a=np.asarray(safe).reshape(-1,3)
    bounds={name:[float(safe_a[:,a].min()),float(safe_a[:,a].max())] if len(safe) else [0.,0.]
            for a,name in enumerate(('vx','vy','wz'))}
    # Isolated step responses and stopping distance, with the same seeded batteries.
    response=[[],[],[]]; stops=[]
    for a,c in enumerate(((.3,0,0),(0,.1,0),(0,0,1.))):
        for seed in range(seeds):
            p,v,f=rollout(sim,seed,[((0,0,0),.5),(c,3.),((0,0,0),2.)])
            if not f:
                response[a].append(v[:175])
                if a==0:
                    stop_idx=175
                    delta=p[stop_idx:,:2]-p[stop_idx,:2]
                    forward=delta@np.array([math.cos(p[stop_idx,2]),math.sin(p[stop_idx,2])])
                    stops.append(float(np.max(forward)))
    # Straight commanded forward endpoint spread (cross-track in the initial frame).
    straight=[r for r in rows if r['safe_observed'] and r['command'][0]>0 and r['command'][1:]==[0.,0.]]
    lateral_sigma=max((r['end_std'][1] for r in straight),default=0.)
    md5=sim.policy_entry['md5']
    envelope={'policy_key':key,'policy_md5':md5,'dynamics':dynamics,'dynamics_version':sim.provenance()['dynamics_version'],**bounds,'safe_cmds':safe,
              'lateral_sigma':lateral_sigma,'stop_dist':{'command_vx':.3,'n':len(stops),
                  'mean':float(np.mean(stops)) if stops else None,'max':max(stops) if stops else None},
              'step_over_height':0.,'fol':fit_fol(response,(.3,.1,1.)),
              'measurement':{'scene':'flat','seeds':seeds,'seconds_per_command':seconds,
                  'n_commands':len(commands),'sampled_commands':[list(c) for c in commands],
                  'safe_rule':'zero observed falls AND useful signed response on every commanded axis; zero retained if fall-free',
                  'limits':'Nominal training structure; battery/sag seeded; no mass/CoM/friction/IMU/encoder/noise DR; no step-over experiment (0 conservative).',
                  'rows':rows}}
    (ROOT/'data'/f'envelope_{key}.json').write_text(json.dumps(envelope,indent=2,allow_nan=False)+'\n')
    primitives=measure_primitives(sim,envelope,primitive_seeds)
    (ROOT/'data'/f'primitives_{key}.json').write_text(json.dumps(primitives,indent=2,allow_nan=False)+'\n')
    forward=[r for r in straight if r['mean_velocity'][0]>.03]
    idle=next((r for r in rows if r['command']==[0.,0.,0.]),None)
    result={'key':key,'policy_md5':md5,'dynamics':dynamics,
            'dynamics_version':sim.provenance()['dynamics_version'],'safe_commands':len(safe),
            'grid_falls':sum(r['falls'] for r in rows),'grid_trials':len(rows)*seeds,
            'grid_fall_rate':sum(r['falls'] for r in rows)/(len(rows)*seeds),
            'forward_safe_commands':len(forward),'max_forward_speed':max((r['mean_velocity'][0] for r in forward),default=0.),
            'idle':idle,'primitive_count':len(primitives['primitives']),
            'usable_primitives':sum(p['fall_rate']<=.05 for p in primitives['primitives']),
            'wall_s':time.time()-start}
    print('COMPLETE',json.dumps(result),flush=True)
    return result

def primitive_commands(envelope):
    """Select measured safe commands nearest a compact locomotion action set."""
    safe=np.asarray(envelope['safe_cmds']).reshape(-1,3)
    targets=((.3,0,0),(-.3,0,0),(0,.1,0),(0,-.1,0),(0,0,1.),(0,0,-1.),(.3,0,.5),(.3,0,-.5))
    out=[]
    for target in targets:
        if not len(safe): break
        # Require the same active axes/signs; never substitute a standstill primitive.
        tar=np.asarray(target); active=np.abs(tar)>1e-9
        mask=np.all(np.abs(safe[:,~active])<1e-9,axis=1)&np.all(safe[:,active]*tar[active]>0,axis=1)
        candidates=safe[mask]
        if len(candidates):
            c=candidates[np.argmin(np.linalg.norm((candidates-tar)/[.3,.2,1.],axis=1))].tolist()
            if c not in out: out.append(c)
    return out

def measure_primitives(sim,envelope,seeds):
    ps=[]
    for h in range(16):
        heading=h*2*math.pi/16
        for ci,cmd in enumerate(primitive_commands(envelope)):
            for duration in (.6,1.2):
                traces=[]; ends=[]; falls=0
                for seed in range(seeds):
                    p,v,f=rollout(sim,seed,[(cmd,duration)],heading)
                    falls+=f; ends.append(p[-1]);
                    if not f: traces.append(p)
                end=np.asarray(ends); cov=np.cov(end.T,ddof=1) if seeds>1 else np.zeros((3,3))
                mean_trace=np.mean(traces,0) if traces else np.array([[0,0,0],end.mean(0)])
                ps.append({'id':f'h{h}_c{ci}_d{duration}','start_heading_idx':h,
                    'cmd_sequence':[[*cmd,duration]],'end_mean':end.mean(0).tolist(),'end_cov':cov.tolist(),
                    'duration':duration,'fall_rate':falls/seeds,'trials':seeds,
                    'fall_rate_wilson95_upper':wilson_upper(falls,seeds),
                    'swept':mean_trace[::max(1,len(mean_trace)//30),:2].tolist()+[mean_trace[-1,:2].tolist()]})
    return {'policy_key':sim.policy_key,'policy_md5':sim.policy_entry['md5'],'dynamics':sim.dynamics,'dynamics_version':sim.provenance()['dynamics_version'],
            'n_headings':16,'measurement':{'seeds':seeds,'scene':'flat','frame':'world-aligned start-relative'},'primitives':ps}

def select_walker(results):
    versions={r.get('dynamics_version') for r in results}
    if not results or None in versions or len(versions)!=1:
        raise ValueError(f'Walker results have missing or inconsistent dynamics_version: {versions}')
    # The commanded range alone cannot promote a stationary checkpoint.
    eligible=[r for r in results if r['forward_safe_commands'] and r['usable_primitives']]
    best=min(eligible,key=lambda r:(r['grid_fall_rate'],-r['safe_commands'],-r['max_forward_speed'])) if eligible else None
    return {'selected':best['key'] if best else None,'dynamics':results[0]['dynamics'],
            'dynamics_version':versions.pop(),
            'selection_rule':'Require useful forward progress and usable primitives; lowest grid fall rate, then widest observed safe command set, then forward speed.',
            'gate':'measured-flat-only' if best else 'FAIL','caveat':'Small zero-fall cells have wide Wilson95 bounds (8 seeds: upper 0.324); does not establish rare-fall safety or independent apartment replay.',
            'policies':results}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--policies',nargs='+')
    ap.add_argument('--all-walkers',action='store_true');ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--primitive-seeds',type=int,default=8)
    ap.add_argument('--seconds',type=float,default=4.);ap.add_argument('--dynamics',choices=['infer','training'],default='training')
    args=ap.parse_args(); keys=accepted_walkers() if args.all_walkers else (args.policies or DEFAULT_KEYS)
    if any(k not in accepted_walkers() for k in keys): raise ValueError('Only accepted walker keys can be measured')
    with ProcessPoolExecutor(max_workers=max(1,min(args.workers,3))) as pool:
        futures=[pool.submit(measure_policy,k,args.dynamics,args.seeds,args.seconds,GRID,args.primitive_seeds) for k in keys]
        results=[f.result() for f in futures]
    gate=select_walker(results);(ROOT/'reports'/'walker_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(json.dumps(gate,indent=2))
if __name__=='__main__': main()
