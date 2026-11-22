"""Posthoc pose-estimation errors against recorded GT, without changing any verdict."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nav import checker

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',required=True);ap.add_argument('--traces',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args(); report=json.loads(Path(args.report).read_text()); rows=[]
    for ep,verdict in sorted(report['episodes'].items()):
        if verdict['task_type']!='nav':continue
        with np.load(Path(args.traces)/(ep+'.npz'),allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));qpos=z['qpos'];est=np.asarray(meta['estimates'],float)
            m=checker.load_model(meta['model_path']);b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'trunk_base')
            a=int(m.jnt_qposadr[m.body_jntadr[b]])
            indices=np.rint(est[:,0]/.02).astype(int)
            if not len(est) or np.any(indices<0) or np.any(indices>=len(qpos)):raise RuntimeError('estimate timeline mismatch '+ep)
            err=np.linalg.norm(est[:,1:3]-qpos[indices,a:a+2],axis=1)
            loc=meta.get('localization',{})
            rows.append({'ep':ep,'success':verdict['success'],'final_estimate_error_m':float(err[-1]),'rms_error_m':float(np.sqrt(np.mean(err**2))),'max_error_m':float(err.max()),'checker_final_pos_error_m':verdict.get('pos_err'),'localization':loc})
    def summarize(group):
        return {'n':len(group),'terminal_error_m':{f'p{p}':float(np.percentile([r['final_estimate_error_m'] for r in group],p)) for p in [50,90,95,100]} if group else {},'camera_renders_total':sum(r['localization'].get('camera_renders',0) for r in group)}
    result={'scope':'posthoc recorded estimate-versus-GT diagnostic; scoring unchanged','report':args.report,'groups':{name:summarize([r for r in rows if pred(r)]) for name,pred in [('all',lambda r:True),('successful',lambda r:r['success']),('failed',lambda r:not r['success'])]},'episodes':rows}
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result['groups'],indent=2),flush=True)
if __name__=='__main__':main()
