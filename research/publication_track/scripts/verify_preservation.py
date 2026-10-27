import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parents[1]
manifest=json.loads((R/'evidence/source_manifest_initial.json').read_text())
changed=[]
for p,v in manifest.items():
 f=R.parent/p
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=v['sha256']:changed.append(p)
r={'files_checked':len(manifest),'changed_or_missing':changed,'all_original_files_unchanged':not changed}
(R/'evidence/preservation_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));assert not changed
