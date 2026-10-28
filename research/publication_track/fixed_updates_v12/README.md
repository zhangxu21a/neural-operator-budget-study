# Frozen fixed-update mechanism diagnostic v12

Protocol and engine/analysis SHA-256 frozen before references, labels or model fitting. 20 serial trajectories, two correlated endpoints each, five new paired pools, both initial rates. Primary last iterates; secondary validation-selected models. No extra seeds or tuning. No performance-based exclusions. All up-front target and input costs separately recorded. See protocol.json for estimands, seeds, schedule, stopping and failure policy.

Runtime: existing bundled Python and original project dependencies; launch engine.py. It uses a lock and refuses to overwrite incomplete outputs. A completed job is resumed only from its saved result; incomplete jobs require explicit audit, never silent restart. analyze.py is frozen and performs no manuscript edits. The final editorial review must examine all outcomes and update paper and submission materials; completion.json alone is not editorial completion.
