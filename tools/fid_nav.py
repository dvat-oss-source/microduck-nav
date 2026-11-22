"""Match fid_mjlab_ref.py schedules using the standalone Sim; CPU only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from nav.sim import Sim

def cmd_at(sched,t):
    c=sched[0][1:]
    for row in sched:
        if t+1e-9>=row[0]: c=row[1:]
    return c

def run(spec,dynamics,scene='flat',seed_offset=0):
    n=round(spec['seconds']/.02); envs=spec['envs']; logs=np.zeros((n+1,len(envs),6))
    falls=[];t0=time.time();sim=Sim(scene,'walk',dynamics=dynamics,record=False)
    for i,e in enumerate(envs):
        gt=sim.reset(int(spec.get('seed',0))+seed_offset+i,(0,0,0),extra={'vin':7.4,'vin_drop_gain':.1})
        def row(g): return [*g.pose2,*g.lin_vel_body[:2],g.ang_vel_body[2]]
        logs[0,i]=row(gt); ft=None
        for k in range(n):
            gt,_=sim.step(cmd_at(e['schedule'],k*.02));logs[k+1,i]=row(gt)
            if gt.fallen:
                ft=(k+1)*.02;logs[k+1:,i]=np.nan;break
        falls.append(ft)
    return {'mode':dynamics,'scene':scene,'seed_offset':seed_offset,'provenance':{k:v for k,v in sim.provenance().items() if k!='params'},'seconds':spec['seconds'],'names':[e['name'] for e in envs],
            'num_envs':len(envs),'wall_s':time.time()-t0,'fall_t':falls,
            'log_cols':['x','y','yaw','vx_b','vy_b','wz_b'],'log_hz':50,
            'log':np.where(np.isnan(logs),None,np.round(logs,5)).tolist()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('mode',choices=['infer','training']);p.add_argument('out')
    p.add_argument('--scene',default='flat');p.add_argument('--seed-offset',type=int,default=0)
    a=p.parse_args();r=run(json.load(open(a.spec)),a.mode,a.scene,a.seed_offset);Path(a.out).write_text(json.dumps(r));print('DONE',r['wall_s'])
