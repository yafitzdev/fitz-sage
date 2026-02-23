# tools/governance/

Offline tooling for the governance pipeline — calibration, feature extraction, and evaluation outside the main test suite.

| File | Purpose |
|---|---|
| `calibrate_cascade.py` | Calibrate the governance cascade thresholds |
| `extract_features.py` | Extract features from raw governance evaluation data |
| `eval_pipeline.py` | Run the full governance evaluation pipeline |
| `train_classifier.py` | Train the governance classifier |
| `data/` | Input data and cached artifacts for governance evaluation |

## Usage

```bash
python tools/governance/eval_pipeline.py
python tools/governance/calibrate_cascade.py
```

See `docs/features/governance/` for design docs and `docs/evaluation/` for results.
