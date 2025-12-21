"""
Evaluation script for comparing model predictions against ground truth.
Ground truth is in data/sample_docs/page_splits/850_gt/*.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from .ingestion import ingest_pdf
from .model import PageClassifier, get_tokenizer
from .labeling import LABELS, LABEL_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GTDocument:
    """Ground truth document structure."""
    filename: str
    total_pages: int
    segment_table_pages: List[int]  # 1-indexed
    segment_pages: List[Dict]  # List of {"segment": "XX", "pages": [1, 2, ...]}
    
    def get_page_label(self, page_num: int) -> str:
        """Get the label for a specific page (1-indexed)."""
        if page_num in self.segment_table_pages:
            return "index"
        
        for seg_info in self.segment_pages:
            if page_num in seg_info["pages"]:
                return "segment"
        
        return "other"
    
    def get_page_segment(self, page_num: int) -> Optional[str]:
        """Get segment ID for a page if it's a segment page."""
        for seg_info in self.segment_pages:
            if page_num in seg_info["pages"]:
                return seg_info["segment"]
        return None


def load_ground_truth(gt_dir: str) -> Dict[str, GTDocument]:
    """Load all ground truth files from directory."""
    gt_path = Path(gt_dir)
    documents = {}
    
    for gt_file in gt_path.glob("*.json"):
        with open(gt_file, 'r') as f:
            data = json.load(f)
        
        # Extract filename stem (without .pdf extension)
        filename = data["filename"].replace(".pdf", "")
        
        documents[filename] = GTDocument(
            filename=data["filename"],
            total_pages=data["total_pages"],
            segment_table_pages=data.get("segment_table_pages", []),
            segment_pages=data.get("segment_pages", [])
        )
    
    return documents


def evaluate_predictions(
    predictions: List[Dict],
    ground_truth: GTDocument
) -> Dict:
    """
    Evaluate predictions against ground truth for a single document.
    
    Args:
        predictions: List of {"page_number": int, "label": str, "confidence": float}
        ground_truth: GTDocument object
    
    Returns:
        Dictionary with metrics
    """
    correct = 0
    total = 0
    
    # Per-class metrics
    class_stats = {
        "index": {"tp": 0, "fp": 0, "fn": 0},
        "segment": {"tp": 0, "fp": 0, "fn": 0},
        "other": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    
    errors = []
    
    for pred in predictions:
        page_num = pred["page_number"]
        pred_label = pred["label"]
        gt_label = ground_truth.get_page_label(page_num)
        
        total += 1
        
        if pred_label == gt_label:
            correct += 1
            class_stats[gt_label]["tp"] += 1
        else:
            class_stats[pred_label]["fp"] += 1
            class_stats[gt_label]["fn"] += 1
            errors.append({
                "page": page_num,
                "predicted": pred_label,
                "actual": gt_label,
                "confidence": pred.get("confidence", 0)
            })
        
        confusion[gt_label][pred_label] += 1
    
    # Compute precision, recall, F1 for each class
    class_metrics = {}
    for label in ["index", "segment", "other"]:
        stats = class_stats[label]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }
    
    accuracy = correct / total if total > 0 else 0
    macro_f1 = sum(m["f1"] for m in class_metrics.values()) / 3
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": dict(confusion),
        "errors": errors,
        "total_pages": total,
        "correct": correct
    }


def apply_hybrid_rules(text: str, pred_label: str, confidence: float, probs) -> Dict:
    """Apply rule-based corrections to model predictions."""
    import re
    
    # Strong indicators for index pages
    index_indicators = [
        r"Pos\.\s+Seg\.",
        r"Pos\s+Id\s+Segment\s+Name",
        r"ID\s+Segment\s+Name\s+Req",
        r"EDI\s+levels\s+and\s+Segments",
        r"Transactions?\s+Summary",
        r"LOOP\s+ID\s*[-–]",
        r"Max\.Use\s+Repeat",
    ]
    index_score = sum(1 for p in index_indicators if re.search(p, text, re.IGNORECASE))
    
    # Check for multiple segment listings
    segment_listing = len(re.findall(r"\d+\s+[A-Z]{2,3}\s+[A-Za-z]", text))
    if segment_listing >= 5:
        index_score += 2
    
    # Strong indicators for other pages
    other_indicators = [
        r"^\s*APPENDIX\s*$",
        r"Transaction\s+Example",
        r"Country\s+Specific\s+Requirements",
        r"GLOSSARY",
        r"Preface",
        r"Purpose\s+and\s+Scope",
    ]
    other_score = sum(1 for p in other_indicators if re.search(p, text, re.IGNORECASE | re.MULTILINE))
    
    # Override if strong indicators
    if index_score >= 2 and pred_label != "index":
        return {"label": "index", "confidence": 0.90}
    elif other_score >= 1 and pred_label != "other":
        return {"label": "other", "confidence": 0.90}
    
    return {"label": pred_label, "confidence": confidence}


def run_inference(
    pdf_path: str,
    model: PageClassifier,
    tokenizer,
    device: str = "cuda"
) -> List[Dict]:
    """
    Run inference on a PDF and return predictions.
    """
    import torch
    
    # Ingest PDF
    pages = ingest_pdf(pdf_path, use_ocr=False)
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for page in pages:
            # Tokenize
            encoding = tokenizer(
                page.text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Predict
            preds, probs = model.predict(input_ids, attention_mask)
            
            pred_label = LABEL_NAMES[preds[0].item()]
            confidence = probs[0][preds[0]].item()
            
            predictions.append({
                "page_number": page.page_number + 1,  # Convert to 1-indexed
                "label": pred_label,
                "confidence": confidence
            })
    
    return predictions


def evaluate_model_on_corpus(
    model_path: str,
    pdf_dir: str,
    gt_dir: str,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate a trained model on the entire corpus.
    
    Args:
        model_path: Path to model checkpoint
        pdf_dir: Directory containing PDFs
        gt_dir: Directory containing ground truth JSON files
        device: Device to run inference on
    
    Returns:
        Aggregated evaluation metrics
    """
    import torch
    
    # Load model
    model = PageClassifier.load(model_path, device=device)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer_path = Path(model_path).parent / "tokenizer"
    tokenizer = get_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)
    
    # Load ground truth
    ground_truths = load_ground_truth(gt_dir)
    
    logger.info(f"Loaded {len(ground_truths)} ground truth documents")
    
    # Find matching PDFs
    pdf_path = Path(pdf_dir)
    
    all_results = {}
    aggregated = {
        "total_pages": 0,
        "correct": 0,
        "class_stats": {
            "index": {"tp": 0, "fp": 0, "fn": 0},
            "segment": {"tp": 0, "fp": 0, "fn": 0},
            "other": {"tp": 0, "fp": 0, "fn": 0}
        }
    }
    
    for doc_name, gt in ground_truths.items():
        # Find corresponding PDF
        pdf_file = pdf_path / f"{doc_name}.pdf"
        
        if not pdf_file.exists():
            logger.warning(f"PDF not found for {doc_name}")
            continue
        
        logger.info(f"Evaluating {doc_name}...")
        
        # Run inference
        predictions = run_inference(str(pdf_file), model, tokenizer, device)
        
        # Evaluate
        result = evaluate_predictions(predictions, gt)
        all_results[doc_name] = result
        
        # Aggregate
        aggregated["total_pages"] += result["total_pages"]
        aggregated["correct"] += result["correct"]
        
        for label in ["index", "segment", "other"]:
            for metric in ["tp", "fp", "fn"]:
                # Compute from class_metrics
                pass
        
        logger.info(f"  Accuracy: {result['accuracy']:.3f}, Macro F1: {result['macro_f1']:.3f}")
    
    # Compute aggregated metrics
    overall_accuracy = aggregated["correct"] / aggregated["total_pages"] if aggregated["total_pages"] > 0 else 0
    
    # Compute macro F1 across all documents
    all_f1s = [r["macro_f1"] for r in all_results.values()]
    avg_macro_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
    
    return {
        "overall_accuracy": overall_accuracy,
        "average_macro_f1": avg_macro_f1,
        "total_pages": aggregated["total_pages"],
        "total_correct": aggregated["correct"],
        "num_documents": len(all_results),
        "per_document": all_results
    }


def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Macro F1: {results['average_macro_f1']:.4f}")
    print(f"Total Pages Evaluated: {results['total_pages']}")
    print(f"Documents Evaluated: {results['num_documents']}")
    
    print("\n" + "-" * 70)
    print("Per-Document Results:")
    print("-" * 70)
    
    for doc_name, doc_result in results["per_document"].items():
        print(f"\n{doc_name}:")
        print(f"  Accuracy: {doc_result['accuracy']:.3f}")
        print(f"  Macro F1: {doc_result['macro_f1']:.3f}")
        
        print("  Class Metrics:")
        for label, metrics in doc_result["class_metrics"].items():
            print(f"    {label}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f} "
                  f"(n={metrics['support']})")
        
        if doc_result["errors"]:
            print(f"  Errors ({len(doc_result['errors'])}):")
            for err in doc_result["errors"][:5]:  # Show first 5 errors
                print(f"    Page {err['page']}: predicted={err['predicted']}, "
                      f"actual={err['actual']} (conf={err['confidence']:.2f})")
            if len(doc_result["errors"]) > 5:
                print(f"    ... and {len(doc_result['errors']) - 5} more errors")
