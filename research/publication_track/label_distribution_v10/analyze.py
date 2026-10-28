from pathlib import Path
import os,sys,json,hashlib
O=Path(__file__).resolve().parent;os.environ['MPLCONFIGDIR']=str(O.parent/'development_label_v9/mpl_config')
sys.path.insert(0,'/Volumes/MacShare/Something/research_q1_20261024/new_directions/python_deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.loads((O/'individual.json').read_text());p=json.loads((O/'protocol.json').read_text());d=np.load(O/'paired_labels.npz');assert len(rows)==512 and np.array_equal(d['params'],np.load(O/'parameters.npy'))
errors={}
for key in ['coarse32','phys32']:
 e=np.linalg.norm(d[key].astype('float64')-d['hf'],axis=1)/np.linalg.norm(d['hf'].astype('float64'),axis=1)*100
 assert np.allclose(e,[r[key+'_error']*100 for r in rows],rtol=1e-12,atol=1e-12);errors[key]=e
C=errors['coarse32'];T=errors['phys32'];delta=T-C
rng=np.random.default_rng(2026102802);boot=np.concatenate([delta[rng.integers(0,512,size=(500,512))].mean(axis=1) for _ in range(40)])
stat=lambda x:dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))
s={'coarse':stat(C),'corrected':stat(T),'delta_pp':stat(delta),'paired_mean_bootstrap95_pp':np.quantile(boot,[.025,.975]).tolist(),'improved':int((delta<0).sum()),'worsened':int((delta>0).sum()),'ties':int((delta==0).sum()),'n':512,'improvement_fraction':float((delta<0).mean()),'worst_corrected_indices':np.argsort(T)[-10:][::-1].tolist(),'ci_scope':'IID input resampling within fixed coefficient distribution and solver; not model training uncertainty or familywise inference'}
(O/'summary.json').write_text(json.dumps(s,indent=2))
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
lim=[min(C.min(),T.min())*.8,max(C.max(),T.max())*1.2]
ax[0].plot(lim,lim,'--',color='#555555',linewidth=1,label='Equal label error')
ax[0].scatter(C,T,s=12,alpha=.55,color='#35658c',edgecolors='none');ax[0].set(xscale='log',yscale='log',xlim=lim,ylim=lim,xlabel='Coarse label error (%)',ylabel='Corrected label error (%)',title='512 frozen diagnostic inputs');ax[0].legend(loc='upper left',fontsize=8);ax[0].set_aspect('equal')
ax[1].step(np.sort(delta),np.arange(1,513)/512,where='post',color='#35658c');ax[1].axvline(0,color='#555555',linestyle='--',linewidth=1);ax[1].set(xlabel='Corrected minus coarse error (percentage points)',ylabel='Empirical cumulative proportion',ylim=(0,1.02),title='Paired error-difference distribution');ax[1].grid(alpha=.15)
fig.savefig(O/'paired_distribution.pdf');fig.savefig(O/'paired_distribution.png',dpi=180)
(O/'audit.json').write_text(json.dumps({'all512errors_recomputed':True,'all_inputs_retained':True,'bootstrap_resamples':20000,'reference_outcomes_not_used':True,'input_sha256_verified':hashlib.sha256((O/'parameters.npy').read_bytes()).hexdigest()==p['inputs_sha256']},indent=2))
print(json.dumps(s,indent=2))
