# Complete development-label diagnostic

Protocol was written before solving all 32 validation inputs per cohort. Three prospective cohorts are kept separate. Do not overwrite this completed directory; use a fresh copy for a new run. `run_diagnostic.py` imports the unchanged original Oracle, verifies all 28 frozen source hashes and the three reference assets, and generates 192 labels. `analyze_and_plot.py` independently recomputes all 96 pairs from saved arrays and checks input/HF identity. Python dependencies are the existing project environment; plotting uses NumPy and Matplotlib.

Errors use the projected 129x129 HF grid and scaled float32 archived targets; norm ratios are computed in float64. All inputs are retained. Diagnostic label time is not a new controlled timing benchmark. The six-input historical screen uses different samples and a different observation norm; do not treat their means as direct replications.

No test/OOD evaluation, model training, rate selection or source modification. Numerical target improvements do not identify the causal effect on trained-network errors.
