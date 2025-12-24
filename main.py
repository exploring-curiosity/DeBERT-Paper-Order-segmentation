#!/usr/bin/env python3
"""
X12 EDI Page Classifier

Pipeline (separate steps):
    1. python main.py label              # Generate pseudo-labels
    2. python main.py embed              # Compute DeBERTa embeddings
    3. python main.py train --mode full  # Train classifier
    4. python main.py evaluate           # Evaluate on ground truth

MLflow:
    mlflow ui --port 5000
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_label(args):
    """Generate pseudo-labels using rule-based system."""
    from src.train import label
    label(args.pdf_dir, args.output)


def cmd_embed(args):
    """Compute DeBERTa embeddings."""
    from src.train import embed
    embed(args.pdf_dir, args.labels, args.output)


def cmd_train(args):
    """Train classifier on embeddings."""
    from src.train import train
    train(args.embeddings, args.mode, args.epochs, args.batch_size, args.lr)


def cmd_evaluate(args):
    """Evaluate model against ground truth."""
    from src.eval_gt import evaluate_model_on_corpus, print_evaluation_report
    
    model_path = Path(args.model)
    gt_dir = Path(args.gt_dir)
    pdf_dir = Path(args.pdf_dir)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    logger.info(f"Evaluating: {model_path}")
    logger.info(f"Ground truth: {gt_dir}")
    
    results = evaluate_model_on_corpus(
        model_path=str(model_path),
        pdf_dir=str(pdf_dir),
        gt_dir=str(gt_dir),
        device="cuda" if not args.cpu else "cpu"
    )
    
    print_evaluation_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="EDI Page Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Label
    lbl = subparsers.add_parser("label", help="Generate pseudo-labels")
    lbl.add_argument("--pdf-dir", default="data/sample_docs/x12_specs/850")
    lbl.add_argument("--output", default="data/pseudo_labels/labels.json")
    
    # Embed
    emb = subparsers.add_parser("embed", help="Compute DeBERTa embeddings")
    emb.add_argument("--pdf-dir", default="data/sample_docs/x12_specs/850")
    emb.add_argument("--labels", default="data/pseudo_labels/labels.json")
    emb.add_argument("--output", default="data/pseudo_labels/embeddings.npz")
    
    # Train
    tr = subparsers.add_parser("train", help="Train classifier")
    tr.add_argument("--embeddings", default="data/pseudo_labels/embeddings.npz")
    tr.add_argument("--mode", choices=["full", "lora"], default="full")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=3e-4)
    
    # Evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate on ground truth")
    ev.add_argument("--model", default="models/full/classifier.pt")
    ev.add_argument("--gt-dir", default="data/sample_docs/page_splits_new/850_gt")
    ev.add_argument("--pdf-dir", default="data/sample_docs/x12_specs/850")
    ev.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "label":
        cmd_label(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
