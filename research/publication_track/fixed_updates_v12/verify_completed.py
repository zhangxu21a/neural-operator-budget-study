"""Independent completion audit; reads records and arrays, does not evaluate models."""
from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'formal_mg_v1'))
import core
np,torch=core.np,core.torch
p=json.loads((R/'protocol.json').read_text());h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert h(R/'protocol.json')==(R/'protocol.sha256').read_text().strip();assert all(h(f)==v for f,v in p['sources'].items())
assert (R/'completion.json').exists(),'Not complete; do not infer results from partial runs'
checks=[]
for seed in p['seeds']:
 d=R/'data'/str(seed);m=json.loads((d/'manifest.json').read_text());assert all(h(d/f)==v for f,v in m['files'].items())
 planned=np.load(d/'batch_rows.npy');lengths=np.load(d/'batch_lengths.npy');needed=np.load(d/'needed_ids.npy');params=np.load(d/'parameters.npy')
 assert np.array_equal(params,core.parameters(p['parameter_seed_offset']+seed,8192,'channel'))
 rng=np.random.default_rng(p['batch_seed_offset']+seed)
 for i in range(1400):assert np.array_equal(needed[planned[i,:lengths[i]]],np.unique(rng.integers(0,8192,8)))
 for name in ['inputs.npy','coarse32.npy','phys32.npy']:
  a=np.load(d/name,mmap_mode='r')
  for off in range(0,len(a),128):assert np.isfinite(a[off:off+128]).all()
 yc=np.load(d/'coarse32.npy',mmap_mode='r');yr=np.load(d/'phys32.npy',mmap_mode='r');assert np.array_equal(yc[needed<32],yr[needed<32])
 for rate in p['rates']:
  for arm in p['arms']:
   out=R/'formal'/f'{arm}_lr{rate}_s{seed}';r=json.loads((out/'result.json').read_text());log=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
   for i,u in enumerate(log):assert u['batch_rows']==planned[i,:lengths[i]].tolist() and u['lr']==rate/(1+i/160)
   ck=torch.load(out/'checkpoints.pt',map_location='cpu',weights_only=False)
   for endpoint,v in ck.items():
    for which in ['last','selected']:
     assert all(bool(torch.isfinite(t).all()) for t in v[which].values())
    assert v['selected_step']<=int(endpoint) and v['selected_step']%100==0
   checks.append({'run':out.name,'successful_updates':len(log),'retained_failure':r['failure'],'endpoints':list(ck),'planned_sequence_verified':True,'lr_sequence_verified':True})
observations=[json.loads(s) for s in (R/'load_observations.jsonl').read_text().splitlines()]
audit={'complete_records':len(checks),'all_frozen_hashes_match':True,'all_precomputed_data_hashes_match':True,'planned_input_and_lr_sequences_match':True,'all_data_finite':True,'common_initial_hf_targets_match':True,'runs':checks,'competing_python_observations':sum(bool(o['competing_python']) for o in observations),'scope':'Python load log does not certify whole-host isolation; retain all runs. No new predictions.','editorial_complete':False}
(R/'independent_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
