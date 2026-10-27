"""Read-only final analysis; run only after the frozen pipeline is complete."""
from pathlib import Path
import json
import statistics as st

root = Path(__file__).resolve().parent
status = json.loads((root / 'status.json').read_text())
assert status['stage'] == 'complete', status['stage']
summary = json.loads((root / 'results_summary.json').read_text())
audit = json.loads((root / 'audit.json').read_text())
post = json.loads((root / 'post_validation.json').read_text())
assert post['passed'] and audit['all_budget_caps_pass']
assert all(audit['source_hashes_match'].values())
manifest = json.loads((root / 'formal_manifest.json').read_text())
rows = []
for method, lr, seed, warmup in manifest['jobs']:
    d = root / 'formal' / f'{method}_lr{lr}_seed{seed}_warm{warmup}'
    run = json.loads((d / 'result.json').read_text())
    assert not run['nonfinite_loss'], d.name
    for budget, v in json.loads((d / 'evaluation.json').read_text()).items():
        rows.append(dict(method=method, lr=lr, seed=seed, budget=int(budget),
                         test=v['test']['relative_l2'], ood=v['ood']['relative_l2'],
                         fine=v['counts']['fine'], steps=v['steps_completed'],
                         timings=v['timings'], ready=v['decision_ready']))

def compare(a, b, budget, metric='test'):
    def values(m):
        return {r['seed']: r[metric] for r in rows
                if r['method'] == m and r['lr'] == .008 and r['budget'] == budget}
    x, y = values(a), values(b)
    assert set(x) == set(y) == {11, 22, 33}
    differences = {s: 1 - x[s] / y[s] for s in x}
    return dict(left=a, right=b, budget=budget, metric=metric,
                paired_relative_improvement=st.mean(differences.values()),
                per_seed=differences, wins=sum(v > 0 for v in differences.values()),
                left_mean=st.mean(x.values()), right_mean=st.mean(y.values()))

comparisons = [compare(a, b, budget, metric)
               for budget in (300, 1200) for metric in ('test', 'ood')
               for a, b in [('gradient', 'uniform'), ('gradient', 'field'),
                            ('data', 'phys_only'), ('data', 'hf')]]
costs = {}
for m in ('gradient', 'uniform', 'field', 'data', 'phys_only', 'hf'):
    rr = [r for r in rows if r['method'] == m and r['lr'] == .008 and r['budget'] == 1200]
    assert len(rr) == 3
    costs[m] = dict(fine_mean=st.mean(r['fine'] for r in rr),
                    pool_fraction=st.mean(r['fine'] for r in rr)/2048,
                    steps_mean=st.mean(r['steps'] for r in rr),
                    timings={k:st.mean(r['timings'][k] for r in rr) for k in rr[0]['timings']})
out = dict(comparisons=comparisons, costs=costs,
           formal_trajectories=len(manifest['jobs']), budget_points=len(rows),
           nonfinite_formal=0, all_checks_passed=True,
           note='Three seeds; 5/20 minute checkpoints are paired, not independent repetitions.')
(root/'decision_analysis.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
