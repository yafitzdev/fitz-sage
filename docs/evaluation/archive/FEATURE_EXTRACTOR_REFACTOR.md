# Feature Extractor Refactor — Deferred

Current implementation: `fitz_sage/core/guardrails/feature_extractor.py`

Works for classifier v1 training. These problems should be addressed if the classifier proves valuable and we need to maintain/extend the feature set.

---

## Problems

### 1. Tight coupling via string keys

The extractor hardcodes constraint names (`"insufficient_evidence"`, `"conflict_aware"`) and their internal metadata key names (`"ie_entity_match_found"`, `"ca_numerical_variance_detected"`). If any constraint renames a metadata key, the extractor silently gets `None` instead of failing. No compile-time contract exists.

### 2. No feature schema/manifest

There's no declaration of "these are all features, their types, and valid ranges." The training pipeline must reverse-engineer the schema from code. If a feature is added or renamed, the trained model breaks silently with no error.

### 3. Repetitive constraint extraction pattern

The "if constraint exists: extract N features; else: set fired=None" block repeats 5 times with identical shape. Adding a new constraint means copy-pasting the block and knowing which metadata keys to read.

### 4. Wrong location

Lives in `core/guardrails/` but has intimate knowledge of every constraint's internals AND the retrieval system's `DetectionSummary`. It's a bridge between constraints and the classifier, not a core guardrail primitive.

### 5. Silent failures on DetectionSummary

Uses `getattr(detection_summary, "has_temporal_intent", None)` instead of a typed protocol. Attribute name changes fail silently.

---

## Proposed fix: Constraint-owned feature schemas

Each constraint declares its own features and extraction logic:

```python
class InsufficientEvidenceConstraint:
    @staticmethod
    def feature_schema() -> dict[str, type]:
        return {
            "ie_fired": bool,
            "ie_max_similarity": float,
            "ie_entity_match_found": bool,
            # ...
        }

    # Extraction happens inside apply() — metadata IS the feature dict
```

The central extractor becomes a thin collector:

```python
def extract_features(query, chunks, constraint_results, detection_summary):
    features = {}
    for name, result in constraint_results.items():
        constraint_cls = get_constraint_class(name)
        schema = constraint_cls.feature_schema()
        for key in schema:
            features[key] = result.metadata.get(key)
    # + Tier 2 (query/chunk stats) + Tier 3 (detection)
    return features
```

Benefits:
- Each constraint owns its schema — adding a feature is a one-file change
- Schema is inspectable at runtime (for training pipeline compatibility checks)
- Type validation possible (schema declares expected types)
- Feature manifest auto-generated from all registered constraints

### When to do this

After classifier v1 training proves the approach works. If the classifier doesn't beat the hand-coded rules, this refactor is wasted effort.
