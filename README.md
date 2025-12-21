# X12 EDI Page Classifier

AI-powered page classification system for X12 EDI specification documents (850 Purchase Order). Achieves **98.54% accuracy** on the test corpus using a hybrid approach combining rule-based feature extraction with deep learning.

## Page Categories

1. **Index** - Pages containing segment tables/summaries (Table of Contents)
2. **Segment** - Pages describing specific EDI segments (e.g., ST, BEG, ITD, REF, PO1, etc.)
3. **Other** - Preface, appendix, sample data, glossary, or cover pages

## Key Features

- **Weak Supervision**: No ground truth labels required for training - uses sophisticated rule-based heuristics to generate pseudo-labels
- **Multi-Stage Feature Extraction**: Font analysis, layout detection, content patterns, and segment ID recognition
- **DeBERTa-v3**: State-of-the-art transformer backbone (184M parameters)
- **Positional Heuristics**: Uses document structure ("other" pages typically at start/end)
- **OCR Fallback**: Automatic OCR for scanned PDFs using Tesseract
- **Ground Truth Evaluation**: Comprehensive evaluation against labeled test data

## How It Works

The system uses a **4-phase pipeline** combining rule-based feature extraction with deep learning:

### Phase 1: PDF Ingestion (`src/ingestion.py`)
- **Text Extraction**: PyMuPDF extracts text with font information
- **Scanned Page Detection**: Low text density triggers OCR fallback
- **OCR Fallback**: Tesseract OCR for image-based pages

### Phase 2: Feature Extraction (`src/features.py`)

Extracts rich features from each page:

| Feature Category | Examples |
|-----------------|----------|
| **Font Features** | Max/min/avg font size, header font detection |
| **Content Patterns** | Segment headers, TOC patterns, element tables |
| **Index Indicators** | "Pos. Seg. ID Name", "Transactions Summary", "LOOP ID" |
| **Other Indicators** | "Preface", "APPENDIX", "Transaction Example" |
| **Segment Detection** | 78 known X12 segment IDs (ISA, ST, BEG, PO1, etc.) |

### Phase 3: Rule-Based Labeling (`src/labeling_v2.py`)

Applies hierarchical rules with confidence scoring:

1. **Glossary/Document Control** → OTHER (conf: 0.95)
2. **Preface/Introduction** → OTHER (conf: 0.95)
3. **Appendix/Example pages** → OTHER (conf: 0.95)
4. **Table of Contents** → OTHER (conf: 0.98)
5. **Index header patterns** → INDEX (conf: 0.92)
6. **Summary/Notes headers** → INDEX (conf: 0.95)
7. **Segment table (5+ segment IDs + table pattern)** → INDEX (conf: 0.92)
8. **Segment header with ID** → SEGMENT (conf: 0.98)
9. **Element table** → SEGMENT (conf: 0.95)
10. **Context-based continuation** → SEGMENT (inherits from previous)

**Positional Heuristics**:
- Pages before first index/segment are likely "other"
- Last few pages are often "other" (appendix, glossary)
- Middle "other" pages with weak signals become "segment" if preceded by segment

### Phase 4: Model Training (`src/train.py`)

- **Model**: DeBERTa-v3-base (184M parameters)
- **Loss**: Confidence-weighted cross-entropy
- **Class Balancing**: Automatic weight computation for imbalanced classes
- **Training**: GPU-accelerated with learning rate scheduling

### Phase 5: Evaluation (`src/eval_gt.py`)

- Compare predictions against ground truth JSON files
- Per-document and aggregate accuracy metrics
- Precision, Recall, F1 per class

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PDF Corpus                                │
│        data/sample_docs/x12_specs/850/*.pdf                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Ingestion (src/ingestion.py)                   │
│  - PyMuPDF text extraction with font info                   │
│  - OCR fallback for scanned pages                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Feature Extraction (src/features.py)              │
│  - Font size analysis (max, min, avg, std)                  │
│  - Segment header detection (78 known segment IDs)          │
│  - TOC, loop overview, element table patterns               │
│  - Glossary, preface, appendix detection                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Weak Labeling (src/labeling_v2.py)               │
│  - Hierarchical rule-based classification                   │
│  - Positional heuristics (start/end = other)                │
│  - Context-aware segment continuation                       │
│  - Confidence scoring (0.5 - 0.98)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Training (src/train.py)                        │
│  - DeBERTa-v3-base encoder (184M params)                    │
│  - Confidence-weighted cross-entropy loss                   │
│  - Stratified train/val split                               │
│  - GPU-accelerated training                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation (src/eval_gt.py)                    │
│  - Compare against ground truth                             │
│  - Per-document accuracy and macro F1                       │
│  - Confusion matrix and error analysis                      │
└─────────────────────────────────────────────────────────────┘
```

## Setup

```bash
# Create mamba environment
mamba env create -f environment.yml
mamba activate l2labs

# Or using pip
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU training)
- Transformers 4.35+
- PyMuPDF 1.23+
- Tesseract (optional, for OCR)

## Usage

### Step 1: Prepare Training Data (Weak Supervision)

Generate pseudo-labels from PDFs without using ground truth:

```bash
python main.py prepare --pdf-dir data/sample_docs/x12_specs/850
```

This creates `data/training_data_weak.json` with auto-labeled pages.

### Step 2: Train the Model

Train the DeBERTa-v3 classifier on GPU:

```bash
python main.py train --epochs 10 --batch-size 4
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 4)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--model-name`: HuggingFace model (default: microsoft/deberta-v3-base)
- `--no-confidence-weights`: Disable confidence-weighted loss

### Step 3: Evaluate Against Ground Truth

Evaluate the trained model against labeled test data:

```bash
python main.py evaluate --gt-dir data/sample_docs/page_splits/850_gt
```

### Step 4: Classify a PDF

Classify pages in a new PDF:

```bash
python main.py classify data/sample_docs/x12_specs/850/amerisourcebergen.pdf
```

Options:
- `--model`: Path to model checkpoint (default: models/best_model.pt)
- `--output`: Save results to JSON file
- `--use-ocr`: Enable OCR for scanned pages

## Output Format

The classifier outputs:
- **Page type** (index, segment, other)
- **Confidence score** for each classification
- **Segment name** for segment pages (e.g., "ST - Transaction Set Header")
- **Grouped results** showing multi-page segments and index sections

Example output:
```
Total pages: 25
Index pages: 2
Segment pages: 21
Other pages: 2

--- Index Page Groups ---
  Pages: 1, 2

--- Segment Groups ---
  ST - Transaction Set Header: Pages 3
  BEG - Beginning Segment for Purchase Order: Pages 4
  ITD - Terms of Sale/Deferred Terms of Sale: Pages 5
  REF - Reference Identification: Pages 6, 7, 8
  ...

--- Other Pages ---
  Pages: 24, 25
```

### JSON Output Structure

```json
{
  "total_pages": 25,
  "summary": {
    "index_pages": 2,
    "segment_pages": 21,
    "other_pages": 2
  },
  "index_groups": [
    {"pages": [1, 2], "start_page": 1}
  ],
  "segment_groups": [
    {"segment_name": "ST - Transaction Set Header", "pages": [3], "start_page": 3},
    {"segment_name": "REF - Reference Identification", "pages": [6, 7, 8], "start_page": 6}
  ],
  "other_pages": [24, 25],
  "detailed_classifications": [
    {"page_number": 1, "type": "index", "confidence": 0.95, "segment_name": null},
    {"page_number": 3, "type": "segment", "confidence": 0.95, "segment_name": "ST - Transaction Set Header"}
  ]
}
```

## Extending the Model

### Adding New Training Data

1. Add new PDF files to the `data/` directory
2. Create manual labels in `data/labels.json`:

```json
{
  "0": {"label": "index", "segment_name": ""},
  "1": {"label": "segment", "segment_name": "ST - Transaction Set Header"},
  "2": {"label": "other", "segment_name": ""}
}
```

3. Re-run training:
```bash
python main.py prepare --pdf data/new_document.pdf
python main.py train --epochs 20
```

### Supporting New PDF Formats

The parser supports multiple 850 document formats:

1. **AmerisourceBergen format**: `Segment: XX Segment Name`
2. **ERICO format**: Segment ID on separate line with `Pos:` marker
3. **3M format**: Table-based index with `Pos. Seg. ID Name` columns

To add support for new formats, update the patterns in:
- `src/pdf_parser.py`: `extract_segment_name()` and `has_segment_structure()`
- `src/inference.py`: `_apply_hybrid_rules()`

### Improving Accuracy

1. **Add more training data**: The model improves with more diverse examples
2. **Fine-tune hyperparameters**: Adjust learning rate, epochs, batch size
3. **Update heuristics**: Add patterns for new document formats in `_apply_hybrid_rules()`

## Project Structure

```
L2labs/
├── data/
│   ├── 850---purchase-order-regular-and-drop-ship_updated.pdf
│   ├── ERICO850_4010.pdf
│   ├── YZJMX41HFAVWYD5F0SU6_850-PO_R10.pdf
│   ├── training_data.json
│   └── *_results.json
├── models/
│   ├── best_model.pt          # Best validation accuracy checkpoint
│   ├── final_model.pt         # Final epoch checkpoint
│   └── tokenizer/             # Saved tokenizer files
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py          # PDF text extraction & pattern detection
│   ├── dataset.py             # Dataset classes & auto-labeling
│   ├── model.py               # DistilBERT classifier architecture
│   ├── train.py               # Training loop with GPU support
│   └── inference.py           # Hybrid classification pipeline
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

## API Usage

```python
from src.inference import PageClassifierInference

# Initialize classifier
classifier = PageClassifierInference(
    model_path="models/best_model.pt",
    tokenizer_path="models/tokenizer"
)

# Classify a PDF
results = classifier.classify_pdf_grouped("path/to/document.pdf")

# Access results
print(f"Total pages: {results['total_pages']}")
print(f"Index pages: {results['summary']['index_pages']}")

for group in results['segment_groups']:
    print(f"{group['segment_name']}: Pages {group['pages']}")
```

## Performance

**Overall Accuracy: 98.54%** (745/756 pages correct)

### Test Corpus Results (27 documents)

| Metric | Value |
|--------|-------|
| Total Pages | 756 |
| Correct | 745 |
| Overall Accuracy | 98.54% |
| Average Macro F1 | 0.92 |

### Documents with 100% Accuracy

- adobe, amerisourcebergen, arnecom, d_and_h, dell, dot, ecia
- fisher_scientific, govx, john_deere, matrix, pubnet, ti, volvo
- lowes, railcis, supplier_in_motion (and more)

### Known Edge Cases

The remaining ~1.5% errors occur in:
- Pages with mixed content (partial segment + notes)
- Unusual document formats
- Loop summary pages that resemble segments

## Segment ID Recognition

The system recognizes 78 standard X12 850 segment IDs:

```
ISA, GS, ST, BEG, CUR, REF, PER, TAX, FOB, ITD, DTM, DIS, INC,
SAC, TD5, TD4, TD3, TD1, MAN, PKG, CTT, SE, GE, IEA, N1, N2, N3,
N4, N9, PO1, PO3, PO4, PID, MEA, PWK, PKD, REQ, SLN, QTY, SCH,
LIN, CTP, PAM, CSH, TC2, AMT, TXI, SDQ, SHP, MSG, FA1, FA2, ...
```

## Project Structure

```
L2labs/
├── data/
│   ├── sample_docs/
│   │   ├── x12_specs/850/          # PDF corpus
│   │   └── page_splits/850_gt/     # Ground truth JSON files
│   └── training_data_weak.json     # Generated training data
├── models/
│   ├── best_model.pt               # Best validation checkpoint
│   ├── final_model.pt              # Final epoch checkpoint
│   └── tokenizer/                  # Saved tokenizer
├── predictions/                    # Generated predictions
├── analysis/
│   ├── pdf_analysis.py             # Document analysis tools
│   ├── error_analysis.py           # Error investigation
│   └── test_labeling_v2.py         # Labeling accuracy test
├── src/
│   ├── ingestion.py                # PDF text extraction
│   ├── features.py                 # Feature extraction
│   ├── labeling_v2.py              # Rule-based labeling
│   ├── model.py                    # DeBERTa classifier
│   ├── train.py                    # Training loop
│   ├── eval_gt.py                  # Ground truth evaluation
│   └── inference.py                # Classification pipeline
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md
```

## License

MIT
