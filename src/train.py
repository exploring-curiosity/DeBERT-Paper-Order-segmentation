"""
Training pipeline for EDI page classifier.

Separate steps:
    1. label   - Generate pseudo-labels using rule-based system
    2. embed   - Compute DeBERTa embeddings for all pages
    3. train   - Train classifier on embeddings

No ground truth used for training - GT is only for evaluation.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split

from src.model import EmbeddingClassifier, LoRAClassifier
from src.labeling_rules import label_document, generate_labels

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_MAP = {"index": 0, "segment": 1, "other": 2}
LABEL_NAMES = {0: "index", 1: "segment", 2: "other"}


# =============================================================================
# STEP 1: LABELING
# =============================================================================

def label(
    pdf_dir: str = "data/sample_docs/x12_specs/850",
    output_path: str = "data/pseudo_labels/labels.json"
):
    """
    Generate pseudo-labels using rule-based system.
    
    This is a separate step that can be run independently.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating labels for PDFs in {pdf_dir}")
    labels = generate_labels(pdf_dir, output_path)
    
    # Print summary
    label_counts = {"index": 0, "segment": 0, "other": 0}
    for item in labels:
        label_counts[item["label"]] += 1
    
    logger.info(f"Label distribution:")
    for lbl, count in label_counts.items():
        logger.info(f"  {lbl}: {count} ({count/len(labels)*100:.1f}%)")
    
    return labels


# =============================================================================
# STEP 2: EMBEDDING
# =============================================================================

def embed(
    pdf_dir: str = "data/sample_docs/x12_specs/850",
    labels_path: str = "data/pseudo_labels/labels.json",
    output_path: str = "data/pseudo_labels/embeddings.npz",
    device: str = "cuda"
):
    """
    Compute DeBERTa embeddings for all labeled pages.
    
    This is a separate step that can be run independently.
    Requires labels.json from the label step.
    """
    from src.pdf_parser import parse_pdf
    
    # Load labels
    with open(labels_path) as f:
        pseudo_labels = json.load(f)
    
    logger.info(f"Computing embeddings for {len(pseudo_labels)} pages...")
    
    embeddings, labels = _compute_embeddings(pdf_dir, pseudo_labels, output_path, device)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Labels distribution: {np.bincount(labels)}")
    
    return embeddings, labels


def _compute_embeddings(
    pdf_dir: str,
    pseudo_labels: list,
    output_path: str = None,
    device: str = "cuda"
):
    """Internal function to compute embeddings."""
    from src.pdf_parser import parse_pdf
    
    logger.info("Loading DeBERTa encoder...")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    encoder.to(device)
    encoder.eval()
    
    pdf_path = Path(pdf_dir)
    embeddings = []
    labels = []
    
    # Group labels by PDF
    labels_by_pdf = {}
    for item in pseudo_labels:
        pdf_name = item["pdf"]
        if pdf_name not in labels_by_pdf:
            labels_by_pdf[pdf_name] = []
        labels_by_pdf[pdf_name].append(item)
    
    with torch.no_grad():
        for pdf_name, page_labels in labels_by_pdf.items():
            pdf_file = pdf_path / pdf_name
            if not pdf_file.exists():
                continue
            
            pages = parse_pdf(str(pdf_file))
            
            for i, page in enumerate(pages):
                if i >= len(page_labels):
                    break
                
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
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(embedding[0])
                labels.append(LABEL_MAP[page_labels[i]["label"]])
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    if output_path:
        np.savez(output_path, embeddings=embeddings, labels=labels)
        logger.info(f"Saved embeddings to {output_path}")
    
    return embeddings, labels


# =============================================================================
# STEP 3: TRAINING
# =============================================================================

def train(
    embeddings_path: str = "data/pseudo_labels/embeddings.npz",
    mode: str = "full",
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "cuda",
    experiment_name: str = "edi_page_classifier"
):
    """
    Train classifier on precomputed embeddings.
    
    This is a separate step that can be run independently.
    Requires embeddings.npz from the embed step.
    """
    # Load embeddings
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    logger.info(f"Loaded {len(embeddings)} embeddings")
    
    output_dir = f"models/{mode}"
    hidden_dim = 768 if mode == "full" else 384
    
    model, val_acc = _train_model(
        embeddings, labels,
        output_dir=output_dir,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
        experiment_name=experiment_name
    )
    
    logger.info(f"Training complete. Best validation accuracy: {val_acc:.4f}")
    return model, val_acc


def _get_gpu_metrics():
    """Get GPU utilization and memory usage."""
    if not NVML_AVAILABLE:
        return {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_utilization": util.gpu,
            "gpu_memory_used_mb": mem.used / 1024 / 1024,
        }
    except:
        return {}


def _train_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str = "models/full",
    mode: str = "full",
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 3e-4,
    hidden_dim: int = 768,
    device: str = "cuda",
    experiment_name: str = "edi_page_classifier"
):
    """
    Train classifier on precomputed embeddings.
    
    Args:
        mode: "full" for larger model, "lora" for smaller model
        experiment_name: MLflow experiment name
    """
    
    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    logger.info(f"Mode: {mode}")
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    logger.info(f"Label distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    # Create model based on mode
    input_dim = embeddings.shape[1]
    if mode == "full":
        model = EmbeddingClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    else:  # lora
        model = LoRAClassifier(input_dim=input_dim, hidden_dim=hidden_dim // 2)
    model.to(device)
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        })
    
    best_acc = 0
    best_state = None
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]
        
        total_loss = 0
        for i in range(0, len(X_train_t), batch_size):
            batch_x = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            preds = logits.argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()
        
        # MLflow logging
        if MLFLOW_AVAILABLE:
            metrics = {"loss": total_loss, "val_acc": acc}
            metrics.update(_get_gpu_metrics())
            mlflow.log_metrics(metrics, step=epoch)
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            logger.info(f"Epoch {epoch+1}: loss={total_loss:.4f}, val_acc={acc:.4f} (new best)")
        elif (epoch + 1) % 50 == 0:
            logger.info(f"Epoch {epoch+1}: loss={total_loss:.4f}, val_acc={acc:.4f}")
    
    training_time = time.time() - start_time
    
    # Save best model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.load_state_dict(best_state)
    model.save(str(output_path / "classifier.pt"))
    
    # Final MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({"best_val_acc": best_acc, "training_time_sec": training_time})
        mlflow.log_artifact(str(output_path / "classifier.pt"))
        mlflow.end_run()
    
    logger.info(f"Best validation accuracy: {best_acc:.4f}")
    logger.info(f"Training time: {training_time:.1f}s")
    logger.info(f"Model saved to {output_path / 'classifier.pt'}")
    
    return model, best_acc


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training pipeline")
    subparsers = parser.add_subparsers(dest="command")
    
    # Label command
    lbl = subparsers.add_parser("label", help="Generate pseudo-labels")
    lbl.add_argument("--pdf-dir", default="data/sample_docs/x12_specs/850")
    lbl.add_argument("--output", default="data/pseudo_labels/labels.json")
    
    # Embed command
    emb = subparsers.add_parser("embed", help="Compute embeddings")
    emb.add_argument("--pdf-dir", default="data/sample_docs/x12_specs/850")
    emb.add_argument("--labels", default="data/pseudo_labels/labels.json")
    emb.add_argument("--output", default="data/pseudo_labels/embeddings.npz")
    
    # Train command
    tr = subparsers.add_parser("train", help="Train classifier")
    tr.add_argument("--embeddings", default="data/pseudo_labels/embeddings.npz")
    tr.add_argument("--mode", choices=["full", "lora"], default="full")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=3e-4)
    
    args = parser.parse_args()
    
    if args.command == "label":
        label(args.pdf_dir, args.output)
    elif args.command == "embed":
        embed(args.pdf_dir, args.labels, args.output)
    elif args.command == "train":
        train(args.embeddings, args.mode, args.epochs, args.batch_size, args.lr)
    else:
        parser.print_help()
