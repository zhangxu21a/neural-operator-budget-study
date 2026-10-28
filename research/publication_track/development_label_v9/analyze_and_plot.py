from pathlib import Path
import os,sys,json,hashlib
O=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(O/'mpl_config')
sys.path.insert(0,'/Volumes/MacShare/Something/research_q1_20261024/new_directions/python_deps')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
protocol=json.loads((O/'protocol.json').read_text());rows=json.loads((O/'individual.json').read_text());summary=json.loads((O/'summary.json').read_text())
checks=[]
for c in protocol['cohorts']:
 with np.load(O/(c+'_paired_labels.npz')) as d:
  assert d['params'].shape[0]==32 and len(np.unique(d['params'],axis=0))==32
  ref=np.load(O.parent/c/'reference.npz');assert np.array_equal(d['params'],ref['validation_p']);assert np.array_equal(d['hf'],ref['validation_y'].reshape(32,-1))
  for key in ['coarse32','phys32']:
   err=100*np.linalg.norm(d[key].astype('float64')-d['hf'],axis=1)/np.linalg.norm(d['hf'].astype('float64'),axis=1)
   assert np.isfinite(err).all()
   raw=[r[key+'_relative_l2']*100 for r in rows if r['cohort']==c]
   assert np.allclose(err,raw,rtol=1e-12,atol=1e-12)
   assert abs(err.mean()-summary[c][key]['mean'])<1e-10
  checks.append({'cohort':c,'n':32,'reference_parameters_and_targets_identical':True,'independent_error_recomputation':True})
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(10,4.1),sharey=True,layout='constrained')
for ax,c,title in zip(axs,protocol['cohorts'],['Baseline cohort','Architecture/horizon cohort','Matched-rate cohort']):
 rr=[r for r in rows if r['cohort']==c]
 for i,r in enumerate(rr):
  jitter=(i-15.5)/155
  y=[100*r['coarse32_relative_l2'],100*r['phys32_relative_l2']]
  ax.plot([jitter,1+jitter],y,color='#aaaaaa',alpha=.5,linewidth=.65,zorder=1)
  ax.scatter(jitter,y[0],s=16,color='#35658c',marker='o',zorder=2)
  ax.scatter(1+jitter,y[1],s=18,color='#b07824',marker='s',zorder=2)
 ax.set_xticks([0,1],['Coarse 32','Corrected 32']);ax.set_xlim(-.25,1.25);ax.set_yscale('log');ax.set_ylim(.6,40);ax.set_yticks([1,2,5,10,20,40],[1,2,5,10,20,40]);ax.grid(axis='y',alpha=.2)
 ax.set_title(title+'\n32 existing validation inputs',fontsize=10)
axs[0].set_ylabel('Relative numerical label error (%)\nLog scale; projected HF reference')
fig.savefig(O/'development_label_errors.pdf');fig.savefig(O/'development_label_errors.png',dpi=180)
(O/'audit.json').write_text(json.dumps({'checks':checks,'all_96_inputs_retained':True,'no_test_selection':True,'plot':'All paired points, common logarithmic scale, no confidence intervals'},indent=2))
print('All 96 saved pairs independently recomputed and checked.')
