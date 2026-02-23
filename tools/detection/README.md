# tools/detection/

Training and evaluation scripts for the `DetectionClassifier` — the ML model that gates LLM calls in `DetectionOrchestrator`.

| File | Purpose |
|---|---|
| `train_classifier.py` | Train temporal and comparison classifiers from labelled data |
| `apply_relabels.py` | Apply manual relabelling corrections to the training dataset |

## How it fits in

The classifier (`retrieval/detection/`) decides, without an LLM call, whether a query needs temporal or comparison handling. These tools produce and update the model artifacts that back that decision.

## Usage

```bash
# Train classifiers
python tools/detection/train_classifier.py

# Apply relabels before retraining
python tools/detection/apply_relabels.py
```

See `docs/evaluation/` for benchmark results.
