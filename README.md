# Neural operator learning under finite construction budgets

Research code and saved numerical records accompanying **Buying labels or training longer? Cost and bias in multifidelity neural operator learning under finite computational budgets**, by Xu Zhang and Xianjian Zhang (National University of Defense Technology).

This is a research reproducibility release, not a claim of peer-reviewed acceptance. It retains favorable and unfavorable outcomes. No new training was performed for this release.

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

The 358 model/reference assets are distributed as **47 independent `model_assets_NNN.tar.gz` archives**, about 24–47 MB each (1.72 GB total), rather than one 1.88 GB transfer. This allows individual failed uploads/downloads to be retried. Every unpacked asset was verified against its original SHA-256 hash. These archives are unnecessary for saved-record analysis and are excluded from Git.

See [`asset_download/README_BATCHES.txt`](asset_download/README_BATCHES.txt) and its accompanying part manifest/checksums. Zenodo deposition is still in progress; reserved DOI `10.5281/zenodo.23686142` is **not a verified published download link**. Do not treat an incomplete draft as a complete release.

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
