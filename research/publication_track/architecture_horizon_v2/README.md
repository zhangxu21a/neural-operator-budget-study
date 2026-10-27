# Second-architecture and longer-horizon replication

Started 2026-10-27, Asia/Seoul. User authorized proceeding with the proposed second architecture and longer budget checks; 2025 CAS major-category tier 2 remains the first-submission target, not an acceptance claim.

- DeepONet: coarse32, phys32, online HF, fixed HF32/HF128; five independent paired training pools and optimizer seeds; 180-second cap with a dependent 120-second checkpoint.
- FNO: coarse32, phys32, online HF, fixed HF128; same five new paired training pools/seeds; fresh 600-second runs with dependent 120/180/360 checkpoints. Old selected-model snapshots lack optimizer/RNG state and are not resumed.
- Each architecture/method receives three 60-second validation-only trials (180 seconds per method). FNO rates: .002/.008/.032; DeepONet: .0005/.002/.008 from its previous documented search range. Rate ranges differ; total allocated search time and trial counts match. No retuning at longer endpoints.
- Total: 27 tuning trials, 45 formal trajectories, 130 correlated budget points. Approximate allocated compute: five hours serially on one CPU thread.
- Fixed gain two, fine512, coarse32, common129 sensor/output grid; same frozen MG-preconditioned CG implementation and rtol1e-8; real label computation including setup. Initial32 HF labels common; fixed128 pays for96 more. Model construction, input generation, validation, checkpoint handling and runtime bookkeeping are charged. Shared references, imports, serialization and final tests are separately measured.
- New reference seeds73001/73002/73003 with32/128/64 inputs; OOD first16 Gaussian coefficients multiplied by1.25. New formal seeds1101/1202/1303/1404/1505, parameter seeds92101/92202/92303/92404/92505. These are not additional replicates of the old reference set.

`preflight.py` checks both architectures, gradients, shapes, initial purchases, checkpoints and deadlines. `study_pipeline.py` freezes source hashes before generating the new references, uses serial randomized seed blocks across architectures, monitors competing Python load with full commands, and pauses between runs if another computation is active. Unexpected competing load is flagged and retained, never used to select or delete seeds. No monitor can certify all host or thermal activity.

All imports of new analysis and numerical modules use explicit file paths and distinct module names. No legacy analysis script is executed. `study_analysis.py` reads only this study's records, writes this directory, and appends a new evidence section to the working paper after all runs complete. `baseline_manuscript/` preserves the starting paper. Original numerical sources, old results and prior negative findings are retained. The working paper still requires interpretation review and compilation after automatic section generation.

Runtime: `/Users/root1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B study_pipeline.py`.

Existing completed records are reusable; incomplete directories cause failure and are retained for diagnosis, not overwritten. Do not edit frozen computational/analysis sources after protocol.json is written. `status.json` and per-run `progress.json` show live progress; `completion.json` records computational completion. `build_engine.py` is provenance for the initial source derivation, not a restart command.

Inference limits: one PDE family; unmatched architecture parameter counts; one tuning pool and short tuning horizon; CPU execution only. A 600-second endpoint is longer-horizon evidence, not proof of convergence. All paired intervals are exploratory; five all-same-direction pairs cannot yield an exact two-sided sign-test p<.05. Different seeds and reference sets mean changes relative to v1 cannot be attributed solely to architecture or training time; the new within-trajectory checkpoints provide the horizon comparison.
