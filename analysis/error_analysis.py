"""
Analyze specific error cases from the current model predictions.
"""

import json
import fitz
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest_pdf
from src.model import PageClassifier, get_tokenizer
from src.labeling import LABEL_NAMES, extract_segment_id
import torch


def load_ground_truth(gt_dir: Path) -> dict:
    """Load all ground truth files."""
    gt_data = {}
    for gt_file in gt_dir.glob("*.json"):
        with open(gt_file) as f:
            data = json.load(f)
            gt_data[gt_file.stem] = data
    return gt_data


def get_gt_label(gt: dict, page_num: int) -> tuple:
    """Get ground truth label and segment for a page."""
    if page_num in gt.get("segment_table_pages", []):
        return "index", None
    
    for seg in gt.get("segment_pages", []):
        if page_num in seg["pages"]:
            is_first = page_num == seg["pages"][0]
            return "segment", seg["segment"], is_first
    
    return "other", None, False


def analyze_errors(model_path: str, pdf_dir: Path, gt_dir: Path):
    """Analyze all prediction errors in detail."""
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PageClassifier.load(model_path, device=device)
    model.eval()
    tokenizer = get_tokenizer()
    
    gt_data = load_ground_truth(gt_dir)
    
    errors = {
        "segment_as_other": [],
        "segment_as_index": [],
        "index_as_other": [],
        "index_as_segment": [],
        "other_as_segment": [],
        "other_as_index": [],
        "continuation_errors": []
    }
    
    total_pages = 0
    correct = 0
    
    for stem, gt in gt_data.items():
        pdf_path = pdf_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            continue
        
        # Get predictions
        pages = ingest_pdf(str(pdf_path), use_ocr=False)
        doc = fitz.open(str(pdf_path))
        
        for page in pages:
            page_num = page.page_number + 1
            total_pages += 1
            
            # Get prediction
            encoding = tokenizer(
                page.text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            with torch.no_grad():
                preds, probs = model.predict(
                    encoding["input_ids"].to(device),
                    encoding["attention_mask"].to(device)
                )
            
            pred_label = LABEL_NAMES[preds[0].item()]
            confidence = probs[0][preds[0]].item()
            
            # Get ground truth
            gt_result = get_gt_label(gt, page_num)
            gt_label = gt_result[0]
            gt_segment = gt_result[1] if len(gt_result) > 1 else None
            is_first = gt_result[2] if len(gt_result) > 2 else False
            
            if pred_label == gt_label:
                correct += 1
                continue
            
            # Analyze the error
            fitz_page = doc[page.page_number]
            
            # Get font info
            fonts = []
            blocks = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block["type"] == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span["text"].strip():
                                fonts.append({
                                    "size": span["size"],
                                    "text": span["text"][:50]
                                })
            
            max_font = max([f["size"] for f in fonts]) if fonts else 0
            header_text = " ".join([f["text"] for f in fonts if f["size"] >= max_font * 0.9][:3])
            
            error_info = {
                "file": stem,
                "page": page_num,
                "predicted": pred_label,
                "actual": gt_label,
                "confidence": confidence,
                "segment": gt_segment,
                "is_first_page": is_first if gt_label == "segment" else None,
                "max_font_size": max_font,
                "header_text": header_text[:100],
                "text_preview": page.text[:200].replace("\n", " ")
            }
            
            # Categorize error
            error_key = f"{gt_label}_as_{pred_label}"
            if error_key in errors:
                errors[error_key].append(error_info)
            
            if gt_label == "segment" and not is_first:
                errors["continuation_errors"].append(error_info)
        
        doc.close()
    
    return errors, total_pages, correct


def print_error_analysis(errors: dict, total: int, correct: int):
    """Print detailed error analysis."""
    
    print("=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nTotal pages: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Errors: {total - correct} ({(total-correct)/total*100:.1f}%)")
    
    print("\n" + "-" * 70)
    print("ERROR BREAKDOWN")
    print("-" * 70)
    
    for error_type, error_list in errors.items():
        if error_list:
            print(f"\n### {error_type.upper()} ({len(error_list)} cases) ###")
            
            # Group by file
            by_file = defaultdict(list)
            for e in error_list:
                by_file[e["file"]].append(e)
            
            for file, file_errors in sorted(by_file.items()):
                print(f"\n  {file}:")
                for e in file_errors[:5]:
                    print(f"    Page {e['page']}: pred={e['predicted']}, actual={e['actual']}, conf={e['confidence']:.2f}")
                    if e.get("segment"):
                        print(f"      Segment: {e['segment']}, First page: {e.get('is_first_page')}")
                    print(f"      Max font: {e['max_font_size']:.1f}")
                    print(f"      Header: {e['header_text'][:60]}")
                if len(file_errors) > 5:
                    print(f"    ... and {len(file_errors) - 5} more")
    
    # Summary of patterns
    print("\n" + "=" * 70)
    print("KEY PATTERNS IN ERRORS")
    print("=" * 70)
    
    # Continuation page errors
    cont_errors = errors["continuation_errors"]
    if cont_errors:
        print(f"\n### CONTINUATION PAGE ERRORS ({len(cont_errors)} cases) ###")
        print("These are segment pages that are NOT the first page of the segment")
        
        pred_dist = defaultdict(int)
        for e in cont_errors:
            pred_dist[e["predicted"]] += 1
        print(f"  Predicted as: {dict(pred_dist)}")
        
        # Check font sizes
        font_sizes = [e["max_font_size"] for e in cont_errors]
        if font_sizes:
            print(f"  Avg max font size: {sum(font_sizes)/len(font_sizes):.1f}")
    
    # Segment as other errors (first page)
    first_page_errors = [e for e in errors["segment_as_other"] if e.get("is_first_page")]
    if first_page_errors:
        print(f"\n### FIRST PAGE SEGMENT ERRORS ({len(first_page_errors)} cases) ###")
        for e in first_page_errors[:10]:
            print(f"  {e['file']} p{e['page']}: {e['segment']}")
            print(f"    Header: {e['header_text'][:80]}")


if __name__ == "__main__":
    pdf_dir = Path("data/sample_docs/x12_specs/850")
    gt_dir = Path("data/sample_docs/page_splits/850_gt")
    model_path = "models/best_model.pt"
    
    errors, total, correct = analyze_errors(model_path, pdf_dir, gt_dir)
    print_error_analysis(errors, total, correct)
    
    # Save to file
    output = Path("analysis/error_details.json")
    with open(output, "w") as f:
        json.dump(errors, f, indent=2, default=str)
    print(f"\nDetailed errors saved to {output}")
