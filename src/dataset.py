"""
Dataset classes for training the page classifier.
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .pdf_parser import parse_pdf, PageData


# Class labels
LABELS = {
    "index": 0,
    "segment": 1,
    "other": 2
}

LABEL_NAMES = {v: k for k, v in LABELS.items()}


class PageClassificationDataset(Dataset):
    """Dataset for page classification using text features."""
    
    def __init__(
        self,
        page_data: List[Dict],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 512
    ):
        self.data = page_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = LABELS[item["label"]]
        
        # Tokenize text
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
            "segment_name": item.get("segment_name", "")
        }


def create_training_data_from_pdf(
    pdf_path: str,
    labels_path: Optional[str] = None
) -> List[Dict]:
    """
    Create training data from a PDF file.
    
    If labels_path is provided, load labels from JSON file.
    Otherwise, use heuristics to auto-label pages.
    """
    pages = parse_pdf(pdf_path)
    training_data = []
    
    if labels_path and Path(labels_path).exists():
        # Load manual labels
        with open(labels_path, 'r') as f:
            manual_labels = json.load(f)
        
        for page in pages:
            page_key = str(page.page_number)
            if page_key in manual_labels:
                label_info = manual_labels[page_key]
                training_data.append({
                    "page_number": page.page_number,
                    "text": page.text,
                    "label": label_info["label"],
                    "segment_name": label_info.get("segment_name", "")
                })
    else:
        # Auto-label using heuristics with context from previous pages
        prev_label = None
        prev_segment_name = None
        
        for page in pages:
            label, segment_name = auto_label_page(page, prev_label, prev_segment_name)
            training_data.append({
                "page_number": page.page_number,
                "text": page.text,
                "label": label,
                "segment_name": segment_name
            })
            prev_label = label
            prev_segment_name = segment_name
    
    return training_data


def auto_label_page(
    page: PageData, 
    prev_label: Optional[str] = None,
    prev_segment_name: Optional[str] = None
) -> Tuple[str, str]:
    """
    Automatically label a page based on heuristics.
    Uses context from previous page to handle continuation pages.
    Returns (label, segment_name).
    """
    import re
    text = page.text.lower()
    
    # Check for segment page indicators
    if page.has_segment_header and page.segment_name:
        return "segment", page.segment_name
    
    # Check for index/summary page indicators
    if page.has_table_structure:
        # Additional check: index pages have table headers
        if "page" in text and "pos" in text and "seg" in text:
            return "index", ""
    
    # Check for sample data pages
    if "sample data" in text:
        return "other", ""
    
    # Check for introduction/cover pages
    if "introduction:" in text or "functional group id" in text:
        if page.has_table_structure:
            return "index", ""
        return "other", ""
    
    # Default: if has segment structure, it's a segment page
    if page.has_segment_header:
        return "segment", page.segment_name or ""
    
    # Check for segment continuation page
    # These pages have data element content but no "Segment:" header
    # They typically contain element codes, descriptions, and attributes
    segment_continuation_indicators = [
        r"[A-Z]{2}\d{2}",  # Element references like PO107, ST01
        r"\bAN\s+\d+/\d+",  # Attribute patterns like "AN 1/48"
        r"\bID\s+\d+/\d+",  # ID patterns
        r"\bM\s+AN\s+",  # Mandatory alphanumeric
        r"\bC\s+AN\s+",  # Conditional alphanumeric
        r"product/service",
        r"identifying number",
        r"code\s+uniquely",
    ]
    
    continuation_count = sum(
        1 for pattern in segment_continuation_indicators 
        if re.search(pattern, page.text, re.IGNORECASE)
    )
    
    # If previous page was a segment and this looks like continuation
    if prev_label == "segment" and continuation_count >= 2:
        return "segment", prev_segment_name or ""
    
    return "other", ""


def save_training_data(data: List[Dict], output_path: str):
    """Save training data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_training_data(input_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Generate training data from sample PDF
    pdf_path = "data/850---purchase-order-regular-and-drop-ship_updated.pdf"
    training_data = create_training_data_from_pdf(pdf_path)
    
    # Print summary
    from collections import Counter
    label_counts = Counter(item["label"] for item in training_data)
    print(f"Training data summary:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} pages")
    
    # Print detailed labels
    print("\nDetailed labels:")
    for item in training_data:
        segment_info = f" ({item['segment_name']})" if item['segment_name'] else ""
        print(f"  Page {item['page_number']}: {item['label']}{segment_info}")
    
    # Save training data
    save_training_data(training_data, "data/training_data.json")
    print(f"\nSaved training data to data/training_data.json")
