Model/reference assets in independently extractable batches

These model_assets_NNN.tar.gz archives replace the single model_reference_assets.tar transport file. They contain the same 358 assets with identical SHA-256 hashes. No research records have been removed or changed.

Download every model_assets_*.tar.gz file plus analysis_and_sources.zip into one folder. Extract the analysis ZIP. Extract each asset archive into that SAME folder; all use reviewer_bundle/ paths. Do not concatenate these independent gzip archives.

On macOS/Linux, from the download directory:
  shasum -a 256 -c SHA256SUMS.txt
  for f in model_assets_*.tar.gz; do tar -xzf "$f"; done
  cd reviewer_bundle
  python3 -I tools/reproduce.py --verify-assets --out ../verified_analysis

On Windows, extract every archive to the same destination using 7-Zip, then run the same Python command from reviewer_bundle. asset_parts_manifest.json lists member filenames and compressed-part hashes. Every part is independently retryable. The old single TAR is not required.

Original code is MIT. Numerical data/documentation are CC BY 4.0. Third-party software retains its own licenses.
