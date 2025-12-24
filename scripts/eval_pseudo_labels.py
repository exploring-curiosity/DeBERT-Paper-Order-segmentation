"""
Evaluate pseudo-label quality against ground truth.
This measures how well the rule-based labeling matches GT.
Target: 98%+ accuracy before training ML model.
"""

import json
from pathlib import Path
from src.labeling_rules import label_document

GT_DIR = "data/sample_docs/page_splits_new/850_gt"
PDF_DIR = "data/sample_docs/x12_specs/850"


def load_gt_labels(gt_file: str) -> list:
    """Load ground truth and convert to page-level labels."""
    with open(gt_file) as f:
        gt = json.load(f)
    
    total_pages = gt["total_pages"]
    labels = ["other"] * total_pages
    
    for page_num in gt.get("segment_table_pages", []):
        if 1 <= page_num <= total_pages:
            labels[page_num - 1] = "index"
    
    for seg_info in gt.get("segment_pages", []):
        for page_num in seg_info.get("pages", []):
            if 1 <= page_num <= total_pages:
                labels[page_num - 1] = "segment"
    
    return labels


def evaluate_pseudo_labels():
    """Compare pseudo-labels from rules against ground truth."""
    gt_path = Path(GT_DIR)
    pdf_path = Path(PDF_DIR)
    
    total_correct = 0
    total_pages = 0
    errors_by_doc = {}
    
    for gt_file in sorted(gt_path.glob("*.json")):
        doc_name = gt_file.stem
        pdf_file = pdf_path / f"{doc_name}.pdf"
        
        if not pdf_file.exists():
            print(f"SKIP: {doc_name} - PDF not found")
            continue
        
        # Get ground truth labels
        gt_labels = load_gt_labels(str(gt_file))
        
        # Get pseudo-labels from rules
        pseudo_results = label_document(str(pdf_file))
        pseudo_labels = [r.label for r in pseudo_results]
        
        if len(gt_labels) != len(pseudo_labels):
            print(f"SKIP: {doc_name} - page count mismatch ({len(gt_labels)} vs {len(pseudo_labels)})")
            continue
        
        # Compare
        doc_correct = 0
        doc_errors = []
        for i, (gt, pred) in enumerate(zip(gt_labels, pseudo_labels)):
            if gt == pred:
                doc_correct += 1
            else:
                doc_errors.append({
                    "page": i + 1,
                    "gt": gt,
                    "pred": pred,
                    "reason": pseudo_results[i].reason if i < len(pseudo_results) else ""
                })
        
        doc_acc = doc_correct / len(gt_labels)
        total_correct += doc_correct
        total_pages += len(gt_labels)
        
        if doc_errors:
            errors_by_doc[doc_name] = {
                "accuracy": doc_acc,
                "errors": doc_errors
            }
            print(f"{doc_name}: {doc_acc:.1%} ({len(doc_errors)} errors)")
        else:
            print(f"{doc_name}: 100%")
    
    overall_acc = total_correct / total_pages if total_pages > 0 else 0
    print(f"\n{'='*50}")
    print(f"OVERALL PSEUDO-LABEL ACCURACY: {overall_acc:.2%}")
    print(f"Total: {total_correct}/{total_pages} pages correct")
    print(f"{'='*50}")
    
    # Show detailed errors for worst docs
    print("\nDETAILED ERRORS (worst docs):")
    for doc_name, data in sorted(errors_by_doc.items(), key=lambda x: x[1]["accuracy"]):
        print(f"\n{doc_name} ({data['accuracy']:.1%}):")
        for err in data["errors"][:5]:
            print(f"  Page {err['page']}: GT={err['gt']}, Pred={err['pred']} ({err['reason']})")
    
    return overall_acc, errors_by_doc


if __name__ == "__main__":
    evaluate_pseudo_labels()
