# X12 EDI Page Classifier

ML-based page classification for X12 EDI specification documents using DeBERTa embeddings.

## Quick Start

```bash
# Step 1: Generate pseudo-labels using rule-based system
python main.py label

# Step 2: Compute DeBERTa embeddings  
python main.py embed

# Step 3: Train classifier
python main.py train --mode full

# Step 4: Evaluate on ground truth
python main.py evaluate --model models/full/classifier.pt
```

## Architecture

```
PDF Corpus → Pseudo-Labeling → DeBERTa Embeddings → Classifier → Evaluation
             (labeling_rules.py)   (768-dim)         (3-class)    (vs GT)
```

## Rule-Based Classifier (`src/labeling_rules.py`)

The pseudo-labeling system uses **meaningful structural patterns** to classify pages. Rules are designed to detect what a human would recognize.

### Page Types

| Type | What a Human Sees |
|------|-------------------|
| **INDEX** | A table listing segments with positions (segment directory) |
| **SEGMENT** | Definition of a specific segment with its elements |
| **OTHER** | Front matter (preface, glossary) or back matter (appendix, examples) |

### Classification Rules

#### Rule 1: Front/Back Matter Detection
Detects pages by their distinctive titles:
- **Glossary**, **Preface**, **Introduction** → OTHER
- **Appendix**, **Table of Contents** → OTHER

#### Rule 2: Segment Label Header Detection
Detects explicit segment page markers:
- **Format 1**: `Segment ST – Transaction Set Header` (same line)
- **Format 2**: `Segment` on one line, `TD5 - Carrier Details` on next line
- Handles both en-dash (–) and hyphen (-)

#### Rule 3: Segment Definition Detection
A segment page has:
- **Segment header**: e.g., "BEG - Beginning Segment for Purchase Order"
- **Element definitions**: Element table with data types (M AN 1/30)
- **Definition structure**: Pos:, Max:, Usage: patterns

#### Rule 4: Index Table Detection
An index page has:
- **Table headers**: "Pos Seg ID Name", "Req Des Max"
- **Segment listing rows**: Position numbers + segment IDs in tabular format
- **Multiple segments listed**: Table structure with segment directory

#### Rule 5: Context-Based Continuation
Pages following segments often continue the same segment:
- **Element references**: PO101, BEG02 patterns
- **Syntax/Semantics sections**
- **Code value tables**
- **Description patterns**

#### Rule 6: Index Continuation
Pages following index continue the listing:
- **Segment listing table structure**
- **Multiple segment IDs in text**

#### Rule 7: Loop Overview Detection
Pages that discuss loop structure but aren't segment definitions → OTHER

#### Rule 8: Segment Pattern Detection
Multiple segment documentation patterns indicate segment content:
- Pos:, Max:, Loop:, Usage: patterns
- Element references
- Syntax Rules, Semantics sections

#### Rule 9: Document Structure Validation
Second pass corrects low-confidence labels based on document flow:
- Isolated OTHER pages between segments → likely SEGMENT continuation
- Low-confidence pages near index → likely INDEX continuation

### Key Features Extracted (`src/features.py`)

- **Segment IDs**: 51 known X12 segment codes (ST, BEG, PO1, N1, etc.)
- **Font analysis**: Large fonts indicate headers
- **Pattern matching**: Element tables, TOC patterns, loop overviews
- **Context**: Previous page label for continuation detection

## Results

| Metric | Value |
|--------|-------|
| **Pseudo-label accuracy** | 95.90% (725/756 pages) |

### Known Labeling Errors (31 pages)

| Document | Accuracy | Errors | Main Issues |
|----------|----------|--------|-------------|
| cardinal_glass | 71.0% | 9 | Over-correction in structure validation |
| ti | 81.0% | 4 | Segment pages falling to default |
| erico | 85.1% | 7 | Index/segment confusion |
| volvo | 88.9% | 2 | Segment pages not detected |
| coupa | 90.9% | 3 | Over-correction to segment |
| john_deere | 91.7% | 3 | Segment pages falling to default |
| govx | 95.0% | 1 | Over-correction to segment |
| lowes | 95.5% | 1 | Index not detected |
| adobe | 96.8% | 1 | Segment detected as appendix |

Error patterns:
1. **SEGMENT→default**: Segment pages missing detection patterns (ti, volvo, john_deere)
2. **OTHER→structure_segment**: Over-correction in validation pass (erico, coupa, govx)
3. **INDEX→segment**: Index pages with segment-like headers (erico, coupa)

## Setup

```bash
mamba env create -f environment.yml
mamba activate l2labs
```

## CLI Reference

```bash
python main.py label [--pdf-dir DIR] [--output FILE]
python main.py embed [--pdf-dir DIR] [--labels FILE] [--output FILE]
python main.py train [--embeddings FILE] [--mode full|lora] [--epochs N]
python main.py evaluate [--model FILE] [--gt-dir DIR]
```

## MLflow Tracking

```bash
mlflow ui --port 5000
# View at http://localhost:5000
```
