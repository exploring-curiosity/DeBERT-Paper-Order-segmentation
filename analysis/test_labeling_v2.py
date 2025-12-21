"""
Test the enhanced labeling system against ground truth.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling_v2 import label_document, LABEL_INDEX, LABEL_SEGMENT, LABEL_OTHER


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
            return "segment", seg["segment"]
    
    return "other", None


def evaluate_labeling(pdf_dir: Path, gt_dir: Path):
    """Evaluate labeling accuracy against ground truth."""
    
    gt_data = load_ground_truth(gt_dir)
    
    total = 0
    correct = 0
    
    errors_by_type = defaultdict(list)
    doc_results = {}
    
    for stem, gt in sorted(gt_data.items()):
        pdf_path = pdf_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            continue
        
        # Get predictions
        try:
            results = label_document(str(pdf_path))
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue
        
        doc_correct = 0
        doc_total = gt.get("total_pages", len(results))
        
        for i, result in enumerate(results):
            page_num = i + 1
            total += 1
            
            gt_label, gt_segment = get_gt_label(gt, page_num)
            pred_label = result.label
            
            if pred_label == gt_label:
                correct += 1
                doc_correct += 1
            else:
                error_key = f"{gt_label}_as_{pred_label}"
                errors_by_type[error_key].append({
                    "file": stem,
                    "page": page_num,
                    "predicted": pred_label,
                    "actual": gt_label,
                    "segment": gt_segment,
                    "confidence": result.confidence,
                    "reason": result.reason
                })
        
        doc_acc = doc_correct / doc_total if doc_total > 0 else 0
        doc_results[stem] = doc_acc
        print(f"{stem}: {doc_correct}/{doc_total} ({doc_acc*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("LABELING ACCURACY (vs Ground Truth)")
    print("=" * 70)
    print(f"\nOverall: {correct}/{total} ({correct/total*100:.2f}%)")
    
    # Error breakdown
    print("\n" + "-" * 50)
    print("ERROR BREAKDOWN")
    print("-" * 50)
    
    for error_type, errors in sorted(errors_by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n{error_type}: {len(errors)} cases")
        for e in errors[:5]:
            print(f"  {e['file']} p{e['page']}: reason={e['reason']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Documents with errors
    print("\n" + "-" * 50)
    print("DOCUMENTS NEEDING ATTENTION")
    print("-" * 50)
    
    for stem, acc in sorted(doc_results.items(), key=lambda x: x[1]):
        if acc < 1.0:
            print(f"  {stem}: {acc*100:.1f}%")
    
    return correct, total, errors_by_type


if __name__ == "__main__":
    pdf_dir = Path("data/sample_docs/x12_specs/850")
    gt_dir = Path("data/sample_docs/page_splits/850_gt")
    
    correct, total, errors = evaluate_labeling(pdf_dir, gt_dir)
