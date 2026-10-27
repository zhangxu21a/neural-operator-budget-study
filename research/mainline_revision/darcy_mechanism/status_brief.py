import pathlib,json,time
r=pathlib.Path(__file__).resolve().parent;s=json.loads((r/'status.json').read_text())
if s['stage'] in ['complete','failed']:print(json.dumps(s,ensure_ascii=False));raise SystemExit
stage=s['stage'];name=s.get('current','');d=r/stage/name
age=max(0,time.time()-d.stat().st_mtime) if d.exists() else 0
print(f'{stage} | {name} | 本轮约{age:.0f}秒 | 已完成{len(s["completed"])}项；串行训练继续。')
