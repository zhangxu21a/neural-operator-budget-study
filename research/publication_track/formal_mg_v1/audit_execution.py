"""Post-run observational audit; does not modify frozen experiment or its statistics."""
import pathlib,json,statistics
R=pathlib.Path(__file__).resolve().parent
obs=[json.loads(x) for x in (R/'load_observations.jsonl').read_text().splitlines()]
events=[json.loads(x) for x in (R/'pipeline.log').read_text().splitlines() if x.startswith('{"stage":')]
rows=[]
for e in events:
 phase=e['stage']
 if phase not in ['formal','tuning']:continue
 name=f"{e['arm']}_s{e['seed']}" if phase=='formal' else f"{e['arm']}_lr{e['lr']}"
 p=R/phase/name/'result.json'
 if not p.exists():continue
 r=json.loads(p.read_text());start=e['time_unix'];end=start+r['wall']
 samples=[o for o in obs if start<=o['observed_at']<=end]
 affected=[o for o in samples if o['other_python_load']]
 rows.append({'phase':phase,'run':name,'start':start,'estimated_end':end,'wall':r['wall'],'observations':len(samples),'observed_competing_load_samples':len(affected),'observed_competing_pids':sorted({v['pid'] for o in affected for v in o['other_python_load']}),'monitor_started_after_run':bool(obs and obs[0]['observed_at']>end),'component_sum':sum(r['timings'].values()),'unattributed_wall':r['wall']-sum(r['timings'].values()),'actual_unused_budget':r['budget']-r['wall']})
summary={'scope':'Five-second ps sampling of other Python CPU above 30%; not universal CPU/GPU/thermal isolation. First tuning trial predates monitor and may overlap another task. End timestamps estimated from pipeline start event plus measured wall; export excluded. No runs removed.','runs':rows,'observed_affected_formal':sum(r['phase']=='formal' and r['observed_competing_load_samples']>0 for r in rows),'observed_affected_tuning':sum(r['phase']=='tuning' and r['observed_competing_load_samples']>0 for r in rows),'unmonitored_runs':sum(r['monitor_started_after_run'] for r in rows),'all_components_within_wall':all(r['unattributed_wall']>=-1e-6 for r in rows)}
(R/'execution_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
