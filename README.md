# Neural operator learning under finite construction budgets

Research code and saved numerical records accompanying **Label acquisition and reuse in neural operator learning under finite construction budgets**, by Xu Zhang and Xianjian Zhang (National University of Defense Technology).

This is a research reproducibility release, not a claim of peer-reviewed acceptance. It retains favorable and unfavorable outcomes. No new training was performed during packaging. The v12 update includes the previously completed 20-trajectory fixed-update experiment.

## Reproduce the saved-record analysis

Python 3.10 or newer; no third-party packages needed for this command:

```sh
git clone https://github.com/zhangxu21a/neural-operator-budget-study.git
cd neural-operator-budget-study
python3 -I tools/reproduce.py --out ../budget-study-reanalysis
```

The output directory must not already exist. The command checks file hashes, budget compliance, validation-selected checkpoints and saved samplewise errors; it recomputes group statistics and paired contrasts for **100 formal runs, 48 tuning trials and 280 dependent endpoints** in three separate prospective cohorts. It does not train models or run new predictions. The legacy appendix cohort remains separate and is not pooled with these cohorts.

For figures, install NumPy and Matplotlib in your environment and consult `python3 tools/plot_main.py --help`. Recorded training dependencies are in `training_requirements_recorded.txt`; the optional launcher `tools/replay_model.py` requires explicit opt-in to start training. Cross-machine retraining is not certified.

## Evidence and limitations

- `research/publication_track/formal_mg_v1`: stronger solver and fixed high-fidelity reuse baselines.
- `research/publication_track/architecture_horizon_v2`: separate architecture and construction-horizon comparisons.
- `research/publication_track/matched_rate_v3`: both matched initial learning rates and all five paired seeds.
- `research/publication_track/mechanism_review_v4`: read-only cost, update-opportunity and selected-checkpoint audit.
- `MANIFEST.json`: SHA-256 checksums, distinguishing analysis files from external assets.
- `source_path_map.json`: original frozen source paths mapped to portable archive paths.
- `cost_accounting.json`: separately measured reference generation, export and offline-testing costs.

The findings concern the tested Darcy family and CPU budgets. Correction is not uniformly superior; the lower-rate comparison is inconclusive. The architecture configurations are not capacity matched. Distinct reference datasets are not independent repeats of one experiment. Matching initial rates and wall-time decay does not match update-index learning rates or sample exposure. Purchased labels differ from training exposure, and completed updates differ from the age of validation-selected models. No causal mechanism, convergence or universal allocation rule is established.

## Large model and reference assets

The 358 model/reference assets are retained locally as **47 independent `model_assets_NNN.tar.gz` archives**, about 24–47 MB each (1.72 GB total). Every unpacked asset was verified against its original SHA-256 hash. These archives are unnecessary for saved-record analysis and are excluded from Git; they are not currently publicly hosted.

See [`asset_download/README_BATCHES.txt`](asset_download/README_BATCHES.txt) and its accompanying part manifest/checksums. The Zenodo upload has been stopped and no public Zenodo record is claimed. The corresponding author can provide the retained archives on request; allow time for transfer of the large files.

Once every archive is available, verify and extract them from the repository root:

```sh
shasum -a 256 -c asset_download/SHA256SUMS.txt
for f in model_assets_*.tar.gz; do tar -xzf "$f" --strip-components=1 -C .; done
python3 -I tools/reproduce.py --verify-assets --out ../budget-study-full-verification
```

Do not concatenate these independent archives. If using the original single TAR instead, extract it with `tar -xf /path/to/model_reference_assets.tar --strip-components=1 -C .`; the resulting assets are identical.

Never replace or selectively remove adverse runs. Exact wall-clock training trajectories depend on hardware and scheduling. Archived subdirectory reports preserve their historical state; this README governs the current public release.

## License and provenance

Original code: **MIT**, see `LICENSE` and `LICENSE-CODE.txt`. Numerical data and original documentation: **CC BY 4.0**, see `LICENSE-DATA.txt`. Third-party code retains its original licenses, listed in `THIRD_PARTY_NOTICES.md`.

OpenAI Codex assisted with code preparation, orchestration, analysis and writing. Figures are reproducibly computed from saved records, not image synthesis. Complete historical per-session model versions were not retained. The unpublished manuscript, submission correspondence, third-party publisher full texts and credentials are excluded.


## CMWA v12 update (2026-10-28)

This repository concerns the Darcy neural-operator paper only, not the projected-action PPO paper or other manuscripts. It is not an accepted or published journal article. The manuscript and submission correspondence remain excluded.

New folders under `research/publication_track` retain the complete-development-label diagnostic (v9: 32 inputs per original cohort, 96 total), fresh 512-input diagnostic (v10), and fixed-update FNO comparison (v12: 20 trajectories, 40 dependent endpoints, 160 last/selected and ID/OOD evaluation records). Both rates 0.002/0.008 and all five seeds are retained. These cohorts and endpoints must not be pooled as independent repeats. `revision_sensitivity_v8` retains the saved-input trimming analysis, including reversal of the small low-rate advantage.

Run the additional standard-library-only saved-record audit:

```sh
python3 -I tools/reproduce_v12.py --out ../cmwa-v12-reanalysis
```

This checks frozen source hashes, initial-state identities, batch and per-step learning-rate sequences, saved evaluation means and all 48 paired contrasts (including update and rate interactions); it recomputes the fresh-label mean errors. It never imports the training engine or evaluates a network. `V12_MANIFEST.json` records original file hashes and maps historical absolute source paths to the portable layout. Original scripts are preserved unchanged and may contain machine-specific paths; running them directly is not a certified portable training workflow.

Large v9/v10/v12 arrays and checkpoints are additional to the original 358 assets. Their independent archives and exact member hashes are listed in `asset_download/cmwa_v12_assets_manifest.json`. Extract each archive into the repository root (no strip-components), then optionally add `--verify-assets` to the v12 audit. This is a different extraction convention from the legacy `model_assets_*` archives. Never concatenate independent gzip archives.

The public GitHub release contains lightweight records and code; raw tensors and checkpoints are retained locally and are available from the corresponding author on reasonable request. No public full-asset archive or DOI is claimed. Negative and uncertain results remain: one high-rate short-run corrected target pair worsens, one longer coarse run worsens, and interaction intervals are imprecise. Matched updates do not causally decompose the older wall-clock gaps or establish an optimal HF cache size.

The fixed-update figure can be regenerated with NumPy/Matplotlib installed: `python3 tools/plot_fixed_v12.py --out ../cmwa-v12-figures`. Its PNG was checked pixel-identical to the manuscript figure. Verification outputs are in `verification_cmwa_v12/`.
