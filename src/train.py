"""
Training script for the page classifier.
Uses DeBERTa-v3 with confidence-weighted loss for weak supervision.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import logging

from .model import PageClassifier, DEFAULT_MODEL, get_tokenizer
from .labeling import LABELS, LABEL_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeakSupervisionDataset(Dataset):
    """Dataset for training with weak supervision pseudo-labels."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = LABELS[item["label"]]
        confidence = item.get("confidence", 1.0)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "confidence": torch.tensor(confidence, dtype=torch.float32)
        }


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def compute_class_weights(data: List[Dict]) -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    label_counts = Counter(item["label"] for item in data)
    total = sum(label_counts.values())
    
    weights = []
    for label_name in ["index", "segment", "other"]:
        count = label_counts.get(label_name, 1)
        weight = total / (3 * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    use_confidence_weights: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        confidence = batch["confidence"].to(device) if use_confidence_weights else None
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids, 
            attention_mask,
            labels=labels,
            sample_weights=confidence
        )
        
        loss = outputs["loss"]
        
        # Apply class weights if provided
        if class_weights is not None and not use_confidence_weights:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(outputs["logits"], labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate the model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs["logits"], labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    
    # Compute metrics
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0
    
    # Per-class metrics
    class_metrics = {}
    for label_id, label_name in LABEL_NAMES.items():
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l == label_id)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == label_id and l != label_id)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != label_id and l == label_id)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return {
        "loss": total_loss / len(dataloader) if dataloader else 0,
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }


def train(
    training_data: List[Dict],
    output_dir: str = "models",
    model_name: str = DEFAULT_MODEL,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    val_split: float = 0.15,
    use_confidence_weights: bool = True,
    seed: int = 42
):
    """
    Main training function.
    
    Args:
        training_data: List of dicts with 'text', 'label', 'confidence'
        output_dir: Directory to save models
        model_name: HuggingFace model name
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        val_split: Fraction for validation
        use_confidence_weights: Whether to weight loss by confidence
        seed: Random seed
    """
    torch.manual_seed(seed)
    device = get_device()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Create dataset
    dataset = WeakSupervisionDataset(training_data, tokenizer)
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(dataset)))
    labels = [LABELS[item["label"]] for item in training_data]
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=seed
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = PageClassifier(model_name=model_name)
    model.to(device)
    
    # Class weights
    class_weights = compute_class_weights(training_data)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    best_model_path = output_path / "best_model.pt"
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            class_weights=class_weights,
            use_confidence_weights=use_confidence_weights
        )
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
        logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        
        # Compute macro F1
        macro_f1 = sum(m["f1"] for m in val_metrics["class_metrics"].values()) / 3
        
        for label_name, metrics in val_metrics["class_metrics"].items():
            logger.info(f"  {label_name}: P={metrics['precision']:.3f}, "
                       f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        logger.info(f"  Macro F1: {macro_f1:.3f}")
        
        # Save best model based on macro F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            model.save(best_model_path)
            logger.info(f"Saved best model with macro F1: {best_val_f1:.4f}")
    
    # Save final model
    final_model_path = output_path / "final_model.pt"
    model.save(final_model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Final model saved to: {final_model_path}")
    
    return model


def load_training_data(path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_training_data(data: List[Dict], path: str):
    """Save training data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
