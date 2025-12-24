"""
Evaluate ML model on Ground Truth.
Pure ML inference - no hybrid rules.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

from src.model_v2 import EmbeddingClassifier
from src.pdf_parser import parse_pdf

GT_DIR = "data/sample_docs/page_splits_new/850_gt"
PDF_DIR = "data/sample_docs/x12_specs/850"
MODEL_PATH = "models/v2/classifier.pt"

LABEL_NAMES = {0: "index", 1: "segment", 2: "other"}


def load_gt_labels(gt_file: str) -> list:
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


def evaluate_ml_model(model_path: str = MODEL_PATH, device: str = "cuda"):
    """Evaluate ML model on ground truth - pure ML, no hybrid."""
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = EmbeddingClassifier.load(model_path, device=device)
    model.eval()
    
    # Load encoder for embeddings
    print("Loading DeBERTa encoder...")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    encoder.to(device)
    encoder.eval()
    
    gt_path = Path(GT_DIR)
    pdf_path = Path(PDF_DIR)
    
    total_correct = 0
    total_pages = 0
    errors_by_doc = {}
    
    with torch.no_grad():
        for gt_file in sorted(gt_path.glob("*.json")):
            doc_name = gt_file.stem
            pdf_file = pdf_path / f"{doc_name}.pdf"
            
            if not pdf_file.exists():
                continue
            
            gt_labels = load_gt_labels(str(gt_file))
            pages = parse_pdf(str(pdf_file))
            
            if len(gt_labels) != len(pages):
                print(f"SKIP: {doc_name} - page count mismatch")
                continue
            
            doc_correct = 0
            doc_errors = []
            
            for i, page in enumerate(pages):
                # Get embedding
                encoding = tokenizer(
                    page.text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :]
                
                # Predict
                logits = model(embedding)
                pred_idx = logits.argmax(dim=1).item()
                pred_label = LABEL_NAMES[pred_idx]
                
                gt_label = gt_labels[i]
                
                if pred_label == gt_label:
                    doc_correct += 1
                else:
                    doc_errors.append({
                        "page": i + 1,
                        "gt": gt_label,
                        "pred": pred_label
                    })
            
            doc_acc = doc_correct / len(gt_labels)
            total_correct += doc_correct
            total_pages += len(gt_labels)
            
            if doc_errors:
                errors_by_doc[doc_name] = {"accuracy": doc_acc, "errors": doc_errors}
                print(f"{doc_name}: {doc_acc:.1%} ({len(doc_errors)} errors)")
            else:
                print(f"{doc_name}: 100%")
    
    overall_acc = total_correct / total_pages if total_pages > 0 else 0
    print(f"\n{'='*50}")
    print(f"ML MODEL ACCURACY ON GT: {overall_acc:.2%}")
    print(f"Total: {total_correct}/{total_pages} pages correct")
    print(f"{'='*50}")
    
    return overall_acc, errors_by_doc


if __name__ == "__main__":
    evaluate_ml_model()
