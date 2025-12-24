"""
Improved ML model architecture for page classification.
Uses DeBERTa-v3-base with attention pooling and deeper classifier.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # hidden_states: (batch, seq_len, hidden)
        # attention_mask: (batch, seq_len)
        
        attn_weights = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class PageClassifierV2(nn.Module):
    """
    Improved page classifier with:
    - DeBERTa-v3-base encoder
    - Attention pooling (better than CLS token)
    - Deeper classifier with residual connections
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Attention pooling
        self.pooler = AttentionPooling(self.hidden_size)
        
        # Deeper classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_labels)
        )
        
        self.num_labels = num_labels
        self.model_name = model_name
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Attention pooling
        pooled = self.pooler(hidden_states, attention_mask)
        
        # Classify
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs
    
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "model_name": self.model_name,
                "num_labels": self.num_labels
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(model_name=config["model_name"], num_labels=config["num_labels"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


class EmbeddingClassifier(nn.Module):
    """
    Full classifier for precomputed embeddings.
    Deeper architecture with residual connections.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 768, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_labels)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.mode = "full"
    
    def forward(self, x: torch.Tensor):
        h = self.ln1(self.input_proj(x))
        h = h + self.block1(h)
        h = self.block2(h)
        return self.classifier(h)
    
    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            "mode": self.mode
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_labels=checkpoint["num_labels"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


class LoRAClassifier(nn.Module):
    """
    Lightweight LoRA-style classifier for precomputed embeddings.
    Smaller and faster than full classifier.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 384, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.mode = "lora"
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            "mode": self.mode
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_labels=checkpoint["num_labels"]
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


def load_classifier(path: str, device: str = "cpu"):
    """Load classifier from checkpoint, auto-detecting mode."""
    checkpoint = torch.load(path, map_location=device)
    mode = checkpoint.get("mode", "full")
    
    if mode == "lora":
        return LoRAClassifier.load(path, device)
    else:
        return EmbeddingClassifier.load(path, device)
