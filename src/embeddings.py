"""
Precompute embeddings from transformer to avoid repeated forward passes.
This is the main bottleneck - doing it once saves ~90% training time.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LABEL_TO_ID = {"index": 0, "segment": 1, "other": 2}


class TextDataset(Dataset):
    """Simple dataset for text tokenization."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx][:10000]  # Truncate long texts
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }


def compute_embeddings(
    data_path: str = "data/pseudo_labels/training_data.json",
    model_name: str = "microsoft/deberta-v3-base",
    output_path: str = "data/pseudo_labels/embeddings.npz",
    batch_size: int = 8,
    max_length: int = 512
) -> str:
    """
    Precompute CLS embeddings for all training samples.
    
    Args:
        data_path: Path to pseudo-labels JSON
        model_name: Transformer model name
        output_path: Where to save embeddings
        batch_size: Batch size for inference
        max_length: Max sequence length
    
    Returns:
        Path to saved embeddings file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = np.array([LABEL_TO_ID[item["label"]] for item in data])
    confidences = np.array([item.get("confidence", 1.0) for item in data])
    
    logger.info(f"Loaded {len(texts)} samples")
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Create dataloader
    dataset = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Compute embeddings
    logger.info("Computing embeddings...")
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    
    embeddings = np.vstack(embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_file,
        embeddings=embeddings,
        labels=labels,
        confidences=confidences
    )
    
    logger.info(f"Saved to {output_file}")
    return str(output_file)


def load_embeddings(path: str = "data/pseudo_labels/embeddings.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed embeddings.
    
    Returns:
        (embeddings, labels, confidences)
    """
    data = np.load(path)
    return data["embeddings"], data["labels"], data["confidences"]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument("--data", default="data/pseudo_labels/training_data.json")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--output", default="data/pseudo_labels/embeddings.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    
    args = parser.parse_args()
    compute_embeddings(args.data, args.model, args.output, args.batch_size)
