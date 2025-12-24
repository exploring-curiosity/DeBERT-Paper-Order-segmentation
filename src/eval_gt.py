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

from .pdf_parser import parse_pdf
from transformers import AutoTokenizer

LABELS = ["index", "segment", "other"]
LABEL_NAMES = {0: "index", 1: "segment", 2: "other"}

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
    model,
    tokenizer,
    device: str = "cuda",
    use_classifier_head: bool = False,
    encoder=None
) -> List[Dict]:
    """
    Run ML model inference with document-level hybrid rules.
    Uses label_document for comprehensive rule-based predictions,
    then ML model can correct low-confidence cases.
    """
    import torch
    from src.labeling_rules import label_document
    
    # Get rule-based predictions (document-level context)
    rule_results = label_document(pdf_path)
    
    # Get ML predictions for potential corrections
    pages = parse_pdf(pdf_path)
    ml_predictions = []
    
    model.eval()
    with torch.no_grad():
        for page in pages:
            encoding = tokenizer(
                page.text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            if use_classifier_head:
                encoder_out = encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = encoder_out.last_hidden_state[:, 0, :]
                logits = model(embeddings)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
            else:
                preds, probs = model.predict(input_ids, attention_mask)
            
            ml_predictions.append({
                "label": LABEL_NAMES[preds[0].item()],
                "confidence": probs[0][preds[0]].item()
            })
    
    # Combine: Use rules as primary, ML corrects low-confidence rule predictions
    predictions = []
    for i, rule_result in enumerate(rule_results):
        ml_pred = ml_predictions[i] if i < len(ml_predictions) else None
        
        # Hybrid logic: Trust rules unless they're uncertain AND ML is very confident
        if rule_result.confidence >= 0.75:
            final_label = rule_result.label
            final_conf = rule_result.confidence
        elif ml_pred and ml_pred["confidence"] >= 0.90:
            # Low rule confidence, high ML confidence - use ML
            final_label = ml_pred["label"]
            final_conf = ml_pred["confidence"]
        else:
            # Default to rules
            final_label = rule_result.label
            final_conf = rule_result.confidence
        
        predictions.append({
            "page_number": i + 1,
            "label": final_label,
            "confidence": final_conf
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
    from src.model import load_classifier
    from transformers import AutoModel
    
    model = load_classifier(model_path, device=device)
    model.eval()
    
    # Load encoder for embeddings
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    encoder.to(device)
    encoder.eval()
    
    use_classifier_head = True
    logger.info(f"Loaded {model.mode} classifier")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
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
        predictions = run_inference(
            str(pdf_file), model, tokenizer, device,
            use_classifier_head=use_classifier_head, encoder=encoder
        )
        
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
        
    # Collect all misclassified pages
    all_errors = []
    for doc_name, doc_result in results["per_document"].items():
        for err in doc_result.get("errors", []):
            all_errors.append({
                "document": doc_name,
                "page": err["page"],
                "predicted": err["predicted"],
                "actual": err["actual"],
                "confidence": err.get("confidence", 0.0)
            })
    
    # Print misclassified pages summary
    if all_errors:
        print("\n" + "=" * 70)
        print(f"MISCLASSIFIED PAGES ({len(all_errors)} total)")
        print("=" * 70)
        print(f"{'Document':<35} {'Page':>5} {'Predicted':<10} {'Actual':<10} {'Conf':>6}")
        print("-" * 70)
        
        # Sort by document name, then page number
        all_errors.sort(key=lambda x: (x["document"], x["page"]))
        
        for err in all_errors:
            print(f"{err['document']:<35} {err['page']:>5} {err['predicted']:<10} {err['actual']:<10} {err['confidence']:>6.2f}")
        
        print("-" * 70)
        print(f"Total misclassified: {len(all_errors)} pages")
        
        # Group by error type
        error_types = {}
        for err in all_errors:
            key = f"{err['actual']} → {err['predicted']}"
            error_types[key] = error_types.get(key, 0) + 1
        
        print("\nError breakdown by type:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    else:
        print("\n" + "=" * 70)
        print("NO MISCLASSIFIED PAGES - 100% accuracy!")
        print("=" * 70)
