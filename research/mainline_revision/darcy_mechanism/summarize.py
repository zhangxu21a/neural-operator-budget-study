import pathlib,json,csv,hashlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
p=json.loads((ROOT/'protocol.json').read_text());selected=json.loads((ROOT/'selected.json').read_text())
manifest=json.loads((ROOT/'formal_manifest.json').read_text());allrows=[];groups={};paired={}
def name(m,lr,seed,w):return f'{m}_lr{lr}_seed{seed}_warm{w}'
for m,lr,seed,w in manifest['jobs']:
 d=ROOT/'formal'/name(m,lr,seed,w);cfg=json.loads((d/'result.json').read_text());assert cfg['wall']<=1200
 ev=json.loads((d/'evaluation.json').read_text())
 for budget,v in ev.items():
  assert v['decision_ready']<=int(budget)
  allrows.append({'method':m,'lr':lr,'seed':seed,'warmup':w,'budget':int(budget),'test':v['test']['relative_l2'],'ood':v['ood']['relative_l2'],'fine':v['counts']['fine'],'steps':v['steps_completed'],'scoring_seconds':v['timings']['scoring'],'decision_ready':v['decision_ready']})
for design,methods in [('selected',p['arms']),('matched',p['matched_methods'])]:
 for budget in [300,1200]:
  for m in methods:
   lr=selected[m]['lr'] if design=='selected' else .008
   rr=[r for r in allrows if r['method']==m and r['lr']==lr and r['budget']==budget]
   assert len(rr)==3
   groups[f'{design}_{budget}_{m}']={'n':3,'lr':lr,'test_mean':statistics.mean(r['test'] for r in rr),'test_sd':statistics.stdev(r['test'] for r in rr),'ood_mean':statistics.mean(r['ood'] for r in rr),'fine_mean':statistics.mean(r['fine'] for r in rr),'scoring_mean':statistics.mean(r['scoring_seconds'] for r in rr)}
  for left,right in [('gradient','uniform'),('gradient','field'),('data','hf'),('data','pretrain'),('data','author_mlmc')]:
   if left not in methods or right not in methods:continue
   def vals(m):
    lr=selected[m]['lr'] if design=='selected' else .008
    return {r['seed']:r['test'] for r in allrows if r['method']==m and r['lr']==lr and r['budget']==budget}
   a,b=vals(left),vals(right);v=[1-a[s]/b[s] for s in p['seeds']]
   paired[f'{design}_{budget}_{left}_vs_{right}']={'relative_improvement_mean':statistics.mean(v),'seed_values':v,'wins':sum(x>0 for x in v)}
sourcecheck={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in p['source_hashes'].items()};assert all(sourcecheck.values())
tuning=[json.loads(q.read_text()) for q in (ROOT/'tuning').glob('*/result.json')];assert len(tuning)==48 and all(r['wall']<=180 for r in tuning)
(ROOT/'results_summary.json').write_text(json.dumps({'groups':groups,'paired':paired},indent=2))
(ROOT/'audit.json').write_text(json.dumps({'formal_trajectories':len(manifest['jobs']),'budget_points':len(allrows),'tuning_runs':48,'all_budget_caps_pass':True,'source_hashes_match':sourcecheck},indent=2))
with (ROOT/'all_runs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
labels={'data':'原数据自适应','gradient':'固定率梯度评分','uniform':'固定率随机购买','field':'固定率场误差评分','pretrain':'粗标签预训练后微调','hf':'全高精度','phys_only':'纯物理校正','author_mlmc':'作者 MLMC'}
lines=['# Darcy 数据版收益来源实验','',f'完成48次调参和{len(manifest["jobs"])}条正式轨迹。每条提供5分钟与20分钟两个配对预算点；每组3种子。','', 'gradient、uniform、field 的共同平均购买概率固定为0.25，用于隔离样本间分配。原数据版 data 保留自适应总购买率。所有方法在预算内支付各自实际开销，固定购买概率不等于固定累计标签数。','']
for design in ['selected','matched']:
 lines += ['## '+('等调参额度后选择配置' if design=='selected' else '统一学习率 0.008'),'', '|方法|预算分钟|学习率|测试误差百分比及种子标准差|OOD百分比|高精度标签|评分秒|','|---|---:|---:|---:|---:|---:|---:|']
 for key,g in groups.items():
  if not key.startswith(design+'_'):continue
  _,budget,m=key.split('_',2)
  lines.append(f'|{labels[m]}|{int(budget)//60}|{g["lr"]}|{100*g["test_mean"]:.3f} ± {100*g["test_sd"]:.3f}|{100*g["ood_mean"]:.3f}|{g["fine_mean"]:.1f}|{g["scoring_mean"]:.2f}|')
lines+=['','## 配对机制比较','','正值表示左侧方法误差更低。三种子结果用于检验一致性，不将两个预算点当作独立重复。','','|比较|平均相对改善|改善种子数|逐种子相对改善|','|---|---:|---:|---|']
for k,v in paired.items():lines.append(f'|{k}|{v["relative_improvement_mean"]:.2%}|{v["wins"]}/3|'+', '.join(f'{x:.2%}' for x in v['seed_values'])+'|')
lines+=['','各方法6次调参，每次180秒。预训练组联合搜索3个学习率与2个切换时间，其他组搜索6个学习率；调参总额度均为1080秒。选择配置与统一学习率配置完全相同时复用同一轨迹。所有训练串行单CPU线程，沿用共同32个粗细配对初始化及15秒收尾预留。测试集为新生成的256个ID样本与128个OOD样本，完全不用于选配置。作者代码适配范围沿用上一轮，不冒充原论文全流程复现。']
(ROOT/'results_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'formal_complete':len(manifest['jobs']),'paired':paired},indent=2))
