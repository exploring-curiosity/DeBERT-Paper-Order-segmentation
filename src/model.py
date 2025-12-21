"""
Page Classification Model using DeBERTa-v3.
Supports confidence-weighted training for weak supervision.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional, Dict, List
from pathlib import Path


# Default model - DeBERTa-v3-base is state-of-the-art for text classification
DEFAULT_MODEL = "microsoft/deberta-v3-base"


class PageClassifier(nn.Module):
    """
    DeBERTa-based classifier for X12 EDI document pages.
    
    Classifies pages into:
    - index: Pages with segment tables/summaries
    - segment: Pages describing specific EDI segments
    - other: Design, filler, sample data pages
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_labels: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional confidence-weighted loss.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional class labels
            sample_weights: Optional per-sample weights (e.g., confidence scores)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (index 0)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if sample_weights is not None:
                # Weighted cross-entropy loss
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                per_sample_loss = loss_fn(logits, labels)
                loss = (per_sample_loss * sample_weights).mean()
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }
    
    def predict(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> tuple:
        """Get predictions without computing loss."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        return predictions, probs
    
    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "model_name": self.model_name,
                "num_labels": self.num_labels
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PageClassifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        model = cls(
            model_name=config["model_name"],
            num_labels=config["num_labels"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


def get_tokenizer(model_name_or_path: str = None):
    """Get tokenizer for the model."""
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODEL
    return AutoTokenizer.from_pretrained(model_name_or_path)
