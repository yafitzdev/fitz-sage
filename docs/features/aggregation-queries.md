# Aggregation Queries

Fitz automatically detects and handles aggregation queries - questions asking for lists, counts, or enumerations.

## What It Does

When you ask questions like:
- "List all the test cases that failed"
- "How many errors occurred?"
- "What are the different types of reports?"
- "Enumerate all features of the system"

Fitz detects the aggregation intent and adjusts retrieval to get comprehensive coverage:

1. **Detects aggregation type**: LIST, COUNT, or UNIQUE
2. **Expands retrieval**: Fetches 3-4x more chunks than normal
3. **Augments query**: Adds instructions for exhaustive results
4. **Tags results**: Marks chunks with aggregation metadata for the answer generator

## Aggregation Types

| Type | Triggers | Fetch Multiplier |
|------|----------|------------------|
| **COUNT** | "how many", "count", "number of" | 4x |
| **UNIQUE** | "different", "distinct", "types of" | 3x |
| **LIST** | "list all", "enumerate", "what are the" | 3x |

## How It Works

```
Query: "List all the people mentioned in the documents"
         ↓
   [Aggregation Detector]
         ↓
   Detected: LIST intent
   Target: "people"
   Multiplier: 3x
         ↓
   [Augmented Query]
   "List all the people mentioned in the documents
    (include complete list of all people)"
         ↓
   [Expanded Retrieval]
   Fetch 75 chunks instead of 25
         ↓
   [Results Tagged]
   aggregation_type: LIST
   aggregation_target: people
```

## Example Queries

```
"List all test cases that failed"
→ LIST: fetches 3x chunks to find all failed tests

"How many errors occurred in the authentication module?"
→ COUNT: fetches 4x chunks for accurate counting

"What different types of reports are available?"
→ UNIQUE: fetches 3x chunks to find all report types

"Enumerate all the configuration options"
→ LIST: fetches 3x chunks for comprehensive listing
```

## Technical Details

- **Module**: `fitz_ai/retrieval/aggregation/`
- **Integration**: Baked into `VectorSearchStep` (runs first, before temporal/comparison)
- **No configuration**: Always active, no opt-out

### Detection Patterns

The detector uses regex patterns with confidence scores:

```python
# COUNT (highest priority)
r"\bhow\s+many\b"              # 0.95 confidence
r"\b(count|total|number)\s+(of|the)\b"  # 0.90 confidence

# UNIQUE
r"\b(different|distinct|unique)\s+(types?|kinds?)\b"  # 0.95 confidence
r"\bwhat\s+(types?|kinds?)\s+of\b"  # 0.90 confidence

# LIST
r"\b(list|enumerate|show)\s+(all|every)\b"  # 0.95 confidence
r"\bwhat\s+are\s+(all\s+)?(the\s+)?"  # 0.85 confidence
```

### Query Augmentation

Each aggregation type gets a different augmentation:

- **COUNT**: `"(include all instances for accurate count)"`
- **UNIQUE**: `"(include all different types and categories)"`
- **LIST**: `"(include complete list of all {target})"`
