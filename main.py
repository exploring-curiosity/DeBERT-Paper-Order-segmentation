#!/usr/bin/env python3
"""
X12 EDI Page Classifier - End-to-End Pipeline

Usage:
    # Step 1: Prepare training data using weak supervision (no ground truth)
    python main.py prepare --pdf-dir data/sample_docs/x12_specs/850
    
    # Step 2: Train the model
    python main.py train --epochs 10 --batch-size 4
    
    # Step 3: Evaluate against ground truth
    python main.py evaluate --gt-dir data/sample_docs/page_splits/850_gt
    
    # Step 4: Classify a single PDF
    python main.py classify data/sample_docs/x12_specs/850/amerisourcebergen.pdf
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from collections import Counter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(args):
    """Prepare training data from PDFs using weak supervision."""
    # Use the improved labeling system v2
    from src.labeling_v2 import label_document
    from src.features import extract_document_features
    from src.train import save_training_data
    
    pdf_dir = Path(args.pdf_dir)
    output_path = Path(args.output)
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    training_data = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Use labeling_v2 which includes positional heuristics
            results = label_document(str(pdf_path))
            features_list = extract_document_features(str(pdf_path))
            
            for i, (result, features) in enumerate(zip(results, features_list)):
                training_data.append({
                    "filename": pdf_path.name,
                    "page_number": i + 1,
                    "text": features.text,
                    "label": result.label,
                    "confidence": result.confidence,
                    "segment_id": result.segment_id or "",
                    "reason": result.reason
                })
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    # Save training data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_data(training_data, str(output_path))
    
    # Print summary
    label_counts = Counter(item["label"] for item in training_data)
    logger.info(f"\nTraining data summary ({len(training_data)} total pages):")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(training_data)
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    # Confidence distribution
    avg_conf = sum(item["confidence"] for item in training_data) / len(training_data)
    logger.info(f"\nAverage labeling confidence: {avg_conf:.3f}")
    
    logger.info(f"\nSaved training data to {output_path}")


def train_model(args):
    """Train the classification model."""
    from src.train import train, load_training_data
    from src.model import DEFAULT_MODEL
    
    training_data_path = Path(args.training_data)
    
    if not training_data_path.exists():
        logger.error(f"Training data not found: {training_data_path}")
        logger.error("Run 'python main.py prepare' first to generate training data.")
        sys.exit(1)
    
    logger.info(f"Loading training data from {training_data_path}")
    training_data = load_training_data(str(training_data_path))
    
    # Print summary
    label_counts = Counter(item["label"] for item in training_data)
    logger.info(f"\nTraining data summary ({len(training_data)} pages):")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count}")
    
    # Train
    model_name = args.model_name or DEFAULT_MODEL
    logger.info(f"\nUsing model: {model_name}")
    
    train(
        training_data,
        output_dir=args.output_dir,
        model_name=model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_confidence_weights=not args.no_confidence_weights
    )


def evaluate_model(args):
    """Evaluate model against ground truth."""
    from src.eval_gt import evaluate_model_on_corpus, print_evaluation_report
    
    model_path = Path(args.model)
    gt_dir = Path(args.gt_dir)
    pdf_dir = Path(args.pdf_dir)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Run 'python main.py train' first to train a model.")
        sys.exit(1)
    
    if not gt_dir.exists():
        logger.error(f"Ground truth directory not found: {gt_dir}")
        sys.exit(1)
    
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Ground truth: {gt_dir}")
    logger.info(f"PDFs: {pdf_dir}")
    
    device = "cuda" if not args.cpu else "cpu"
    
    results = evaluate_model_on_corpus(
        model_path=str(model_path),
        pdf_dir=str(pdf_dir),
        gt_dir=str(gt_dir),
        device=device
    )
    
    print_evaluation_report(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert for JSON serialization
        serializable = {
            "overall_accuracy": results["overall_accuracy"],
            "average_macro_f1": results["average_macro_f1"],
            "total_pages": results["total_pages"],
            "num_documents": results["num_documents"],
            "per_document": {
                doc: {
                    "accuracy": r["accuracy"],
                    "macro_f1": r["macro_f1"],
                    "class_metrics": r["class_metrics"],
                    "num_errors": len(r["errors"])
                }
                for doc, r in results["per_document"].items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"\nSaved evaluation results to {output_path}")


def classify_pdf(args):
    """Classify pages in a single PDF."""
    from src.ingestion import ingest_pdf
    from src.model import PageClassifier, get_tokenizer
    from src.labeling import LABEL_NAMES
    import torch
    
    pdf_path = Path(args.pdf)
    model_path = Path(args.model)
    
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        sys.exit(1)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    model = PageClassifier.load(str(model_path), device=device)
    model.eval()
    
    # Load tokenizer
    tokenizer_path = model_path.parent / "tokenizer"
    tokenizer = get_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)
    
    # Ingest PDF
    logger.info(f"Processing {pdf_path}...")
    pages = ingest_pdf(str(pdf_path), use_ocr=args.use_ocr)
    
    results = []
    
    with torch.no_grad():
        for page in tqdm(pages, desc="Classifying pages"):
            encoding = tokenizer(
                page.text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            preds, probs = model.predict(input_ids, attention_mask)
            
            pred_label = LABEL_NAMES[preds[0].item()]
            confidence = probs[0][preds[0]].item()
            
            results.append({
                "page_number": page.page_number + 1,
                "label": pred_label,
                "confidence": confidence,
                "text_preview": page.text[:100].replace("\n", " ")
            })
    
    # Print results
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"\nFile: {pdf_path.name}")
    print(f"Total pages: {len(results)}")
    
    label_counts = Counter(r["label"] for r in results)
    print("\nPage distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    print("\nDetailed results:")
    for r in results:
        print(f"  Page {r['page_number']:3d}: {r['label']:8s} (conf: {r['confidence']:.3f})")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved results to {output_path}")


def predict_gt_format(args):
    """Generate predictions in ground truth format for PDFs."""
    from src.ingestion import ingest_pdf
    from src.model import PageClassifier, get_tokenizer
    from src.labeling import LABEL_NAMES, extract_segment_id
    import torch
    
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Collect PDF files
    input_path = Path(args.input)
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = list(input_path.glob("*.pdf"))
    else:
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_path}")
        sys.exit(1)
    
    logger.info(f"Processing {len(pdf_files)} PDF files")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    model = PageClassifier.load(str(model_path), device=device)
    model.eval()
    
    tokenizer_path = model_path.parent / "tokenizer"
    tokenizer = get_tokenizer(str(tokenizer_path) if tokenizer_path.exists() else None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Ingest and classify
            pages = ingest_pdf(str(pdf_path), use_ocr=args.use_ocr)
            
            predictions = []
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
                    
                    preds, probs = model.predict(input_ids, attention_mask)
                    pred_label = LABEL_NAMES[preds[0].item()]
                    
                    # Try to extract segment ID for segment pages
                    segment_id = None
                    if pred_label == "segment":
                        segment_id = extract_segment_id(page.text)
                    
                    predictions.append({
                        "page": page.page_number + 1,
                        "label": pred_label,
                        "segment_id": segment_id
                    })
            
            # Convert to ground truth format
            gt_output = convert_to_gt_format(
                filename=pdf_path.name,
                total_pages=len(pages),
                predictions=predictions
            )
            
            # Save
            output_file = output_dir / f"{pdf_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(gt_output, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    logger.info(f"\nSaved predictions to {output_dir}")


def convert_to_gt_format(filename: str, total_pages: int, predictions: list) -> dict:
    """Convert predictions to ground truth format."""
    # Collect index pages
    segment_table_pages = [
        p["page"] for p in predictions if p["label"] == "index"
    ]
    
    # Group segment pages by segment ID
    segment_groups = {}
    current_segment = None
    
    for p in predictions:
        if p["label"] == "segment":
            seg_id = p["segment_id"]
            
            # If we have a new segment ID, start a new group
            if seg_id and seg_id != current_segment:
                current_segment = seg_id
                if seg_id not in segment_groups:
                    segment_groups[seg_id] = []
            
            # Add page to current segment group
            if current_segment:
                if current_segment not in segment_groups:
                    segment_groups[current_segment] = []
                segment_groups[current_segment].append(p["page"])
            else:
                # Unknown segment, create placeholder
                unknown_key = f"UNKNOWN_{p['page']}"
                segment_groups[unknown_key] = [p["page"]]
    
    # Convert to list format
    segment_pages = []
    for seg_id, pages in segment_groups.items():
        if not seg_id.startswith("UNKNOWN_"):
            segment_pages.append({
                "segment": seg_id,
                "pages": sorted(set(pages))
            })
    
    # Sort by first page
    segment_pages.sort(key=lambda x: x["pages"][0] if x["pages"] else 0)
    
    return {
        "filename": filename,
        "total_pages": total_pages,
        "segment_table_pages": segment_table_pages,
        "segment_pages": segment_pages
    }


def main():
    parser = argparse.ArgumentParser(
        description="X12 EDI Page Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", 
        help="Prepare training data using weak supervision"
    )
    prepare_parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/sample_docs/x12_specs/850",
        help="Directory containing PDF files"
    )
    prepare_parser.add_argument(
        "--output",
        type=str,
        default="data/training_data_weak.json",
        help="Output path for training data JSON"
    )
    prepare_parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR for scanned pages"
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        "train", 
        help="Train the classification model"
    )
    train_parser.add_argument(
        "--training-data",
        type=str,
        default="data/training_data_weak.json",
        help="Path to training data JSON"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models"
    )
    train_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="HuggingFace model name (default: microsoft/deberta-v3-base)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--no-confidence-weights",
        action="store_true",
        help="Disable confidence-weighted loss"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate model against ground truth"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--gt-dir",
        type=str,
        default="data/sample_docs/page_splits/850_gt",
        help="Directory containing ground truth JSON files"
    )
    eval_parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/sample_docs/x12_specs/850",
        help="Directory containing PDF files"
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    eval_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference"
    )
    
    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", 
        help="Classify pages in a single PDF"
    )
    classify_parser.add_argument(
        "pdf",
        type=str,
        help="Path to PDF file"
    )
    classify_parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    classify_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    classify_parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR for scanned pages"
    )
    classify_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference"
    )
    
    # Predict command (ground truth format output)
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Generate predictions in ground truth format"
    )
    predict_parser.add_argument(
        "input",
        type=str,
        help="Path to PDF file or directory of PDFs"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory for prediction JSON files"
    )
    predict_parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR for scanned pages"
    )
    predict_parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference"
    )
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "classify":
        classify_pdf(args)
    elif args.command == "predict":
        predict_gt_format(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
