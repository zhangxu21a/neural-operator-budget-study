# Frozen prospective efficient-solver budget study

2026-10-26. Target: strengthen manuscript for a first submission to a 2025 CAS major-category tier-two journal; no acceptance claim.

Seven arms: coarse8/coarse32, phys8/phys32, hf_online, hf32/hf128. Five paired, independently sampled training pools and optimization seeds. All label generation uses real numerical computation. HF uses geometric-MG-preconditioned CG including setup. The numerical problem fixes gain2 and fine512, and varies coarse8/32 only. Outputs are projected to129×129. HF arms do not depend on coarse grid and are not duplicated.

Each arm gets three60s learning-rate trials (180s total), followed by five180s formal trajectories with120s checkpoints. This gives21 tuning trials and35 formal trajectories; checkpoints are correlated. Endpoints are short-budget results, not20-minute convergence. Selection never reads tests. A fixed8192 pool and lazy inputs avoid common saturation and huge uncharged input preprocessing. Each arm pays its own actual setup, label, optimization and validation costs. Shared references, imports, final export and testing are measured separately.

Initial32HF labels are common. Fixed128 purchases96 more. Coarse/corrected arms use the initialHF targets if sampled. HF32/128 only sample the purchased subset and otherwise train longer. Three rates .002/.008/.032 were chosen from the existing search range before new results. Short-tuning bias remains disclosed. No test-based method/configuration selection, seed deletion or automated retries of failed scientific runs.

`protocol.json` freezes source hashes at execution; `status.json` is live progress. `pipeline.log` retains execution. An advisory lock prevents a second copy, and active competing Python computation is checked between timed runs. Other processes cannot be universally prevented; single-thread serial methodology remains the scope. Incomplete directories are retained and require explicit archival before rerun.

`pipeline.py` generates references, validates solver agreement and deadlines, tunes, completes all formal runs, analyzes paired effects, and inserts the result section and figure into the manuscript. Completion still requires review of all adverse results, a fresh source/preservation audit, and final compilation. Do not edit computational sources after protocol is written.

## Postprocessing correction

The frozen pipeline imported a legacy module named `analyze` because its path was inserted by legacy runtime setup. Its `complete` status confirms the training loop, not that the intended report existed. The legacy analyzer recomputed 84 derived files; this incident is documented in `analysis_import_incident.json`. Use `python3 -B run_completed_analysis.py` to reproduce the intended analysis by explicit file path. All training results are complete; do not rerun the original pipeline merely to regenerate reports. The frozen computational and analysis sources are retained unchanged. `finalize_manuscript.py` is a one-shot editorial update, not a training or selection stage.
