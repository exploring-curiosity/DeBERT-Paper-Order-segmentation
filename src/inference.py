"""
Inference module for classifying 850 Purchase Order PDF pages.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

from .pdf_parser import parse_pdf, PageData
from .model import PageClassifier
from .dataset import LABELS, LABEL_NAMES


@dataclass
class PageClassification:
    """Classification result for a single page."""
    page_number: int
    page_type: str  # "index", "segment", or "other"
    confidence: float
    segment_name: Optional[str]  # Only for segment pages
    text_preview: str


class PageClassifierInference:
    """Inference class for page classification."""
    
    def __init__(
        self,
        model_path: str = "models/best_model.pt",
        tokenizer_path: str = "models/tokenizer",
        device: Optional[str] = None
    ):
        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint["config"]
        
        self.model = PageClassifier(
            model_name=config["model_name"],
            num_labels=config["num_labels"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = 512
    
    def classify_page(self, text: str, use_hybrid: bool = True) -> Dict:
        """
        Classify a single page from its text content.
        
        Args:
            text: Page text content
            use_hybrid: If True, combine model predictions with rule-based heuristics
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Predict
        predictions, probabilities = self.model.predict(input_ids, attention_mask)
        
        pred_label = predictions[0].item()
        confidence = probabilities[0][pred_label].item()
        
        model_result = {
            "label": LABEL_NAMES[pred_label],
            "confidence": confidence,
            "probabilities": {
                LABEL_NAMES[i]: probabilities[0][i].item()
                for i in range(len(LABEL_NAMES))
            }
        }
        
        if use_hybrid:
            # Apply rule-based corrections for edge cases
            return self._apply_hybrid_rules(text, model_result)
        
        return model_result
    
    def _apply_hybrid_rules(self, text: str, model_result: Dict) -> Dict:
        """Apply rule-based corrections to model predictions."""
        import re
        text_lower = text.lower()
        
        # Check for nearly empty pages (just headers/footers)
        # These are typically "other" pages
        word_count = len(text.split())
        if word_count < 30:
            return {
                "label": "other",
                "confidence": 0.90,
                "probabilities": model_result["probabilities"],
                "rule_override": True,
                "reason": "nearly_empty_page"
            }
        
        # Strong indicators for index pages (multiple formats)
        index_indicators = [
            r"Page\s+Pos\.\s+Seg\.",
            r"No\.\s+No\.\s+ID\s+Name",
            r"Base\s+User",
            r"Max\.Use\s+Repeat",
            r"LOOP ID",
            r"Table of Contents",
            r"Pos\s+Id\s+Segment\s+Name\s+Req",
            r"Functional\s+Group\s*(ID)?\s*=\s*PO",
            # 3M format
            r"Pos\.\s+Seg\.\s+.*Req\.\s+.*Max\.Use",
            r"Heading:\s+\n\s+\n\s+Pos\.",
            r"Summary:\s+\n\s+\n\s+Pos\.",
            r"LOOP ID\s*-\s*[A-Z0-9]+\s+\d+",
            # Additional index patterns
            r"EDI\s+levels\s+and\s+Segments",
            r"Transactions?\s+Summary",
            r"Pos\s+Seg\s+Name\s+Req\s+Max",
            r"ID\s+Segment\s+Name\s+Req\s+Max",
        ]
        index_score = sum(1 for p in index_indicators if re.search(p, text, re.IGNORECASE))
        
        # Check for multiple segment listings (strong index indicator)
        segment_listing_pattern = r"\d+\s+[A-Z]{2,3}\s+[A-Za-z]"
        segment_listings = len(re.findall(segment_listing_pattern, text))
        if segment_listings >= 5:
            index_score += 2
        
        # Strong indicators for segment pages (multiple formats)
        segment_indicators = [
            r"Segment:\s+[A-Z0-9]+\s+[A-Za-z]",
            r"Position:\s+\d+",
            r"Level:\s+(Heading|Detail|Summary)",
            r"Usage:\s+(Mandatory|Optional|Conditional)",
            r"Data Element Summary",
            # ERICO format
            r"Pos:\s*\d+",
            r"Max:\s*\d+",
            r"Element\s+Summary",
            r"User\s+Option\s+\(Usage\)",
            r"(Heading|Detail|Summary)\s*-\s*(Mandatory|Optional)",
        ]
        segment_score = sum(1 for p in segment_indicators if re.search(p, text, re.IGNORECASE))
        
        # Strong indicators for other pages
        other_indicators = [
            r"SAMPLE DATA",
            r"850 REGULAR PURCHASE ORDER",
            r"850 DROP SHIP PURCHASE ORDER",
            r"Preface",
            r"Trading Partner Profile",
            r"Purpose and Scope",
            r"^\s*APPENDIX\s*$",
            r"Transaction\s+Example",
            r"Country\s+Specific\s+Requirements",
            r"GLOSSARY",
            r"Document\s+Control",
            r"Change\s+Record",
        ]
        other_score = sum(1 for p in other_indicators if re.search(p, text, re.IGNORECASE | re.MULTILINE))
        
        # Override model prediction if rule-based score is strong
        if index_score >= 3:
            return {
                "label": "index",
                "confidence": min(0.95, model_result["probabilities"]["index"] + 0.3),
                "probabilities": model_result["probabilities"],
                "rule_override": True
            }
        elif other_score >= 1:
            return {
                "label": "other",
                "confidence": min(0.95, model_result["probabilities"]["other"] + 0.3),
                "probabilities": model_result["probabilities"],
                "rule_override": True
            }
        elif segment_score >= 3:
            return {
                "label": "segment",
                "confidence": min(0.95, model_result["probabilities"]["segment"] + 0.2),
                "probabilities": model_result["probabilities"],
                "rule_override": True
            }
        
        return model_result
    
    def classify_pdf(self, pdf_path: str) -> List[PageClassification]:
        """Classify all pages in a PDF."""
        pages = parse_pdf(pdf_path)
        results = []
        prev_segment_name = None
        
        for page in pages:
            classification = self.classify_page(page.text)
            
            # For segment pages, extract segment name from text
            segment_name = None
            if classification["label"] == "segment":
                segment_name = page.segment_name
                # If no segment name found, use previous segment name (continuation page)
                if not segment_name and prev_segment_name:
                    segment_name = prev_segment_name
                prev_segment_name = segment_name
            else:
                prev_segment_name = None
            
            # Use 1-indexed page numbers for output
            result = PageClassification(
                page_number=page.page_number + 1,  # Convert to 1-indexed
                page_type=classification["label"],
                confidence=classification["confidence"],
                segment_name=segment_name,
                text_preview=page.text[:200].replace("\n", " ")
            )
            results.append(result)
        
        return results
    
    def classify_pdf_grouped(self, pdf_path: str) -> Dict:
        """
        Classify PDF and group results by page type.
        Also handles multi-page segments by grouping consecutive segment pages.
        """
        classifications = self.classify_pdf(pdf_path)
        
        # Group by type
        grouped = {
            "index": [],
            "segment": [],
            "other": []
        }
        
        for cls in classifications:
            grouped[cls.page_type].append({
                "page_number": cls.page_number,  # Already 1-indexed from classify_pdf
                "confidence": cls.confidence,
                "segment_name": cls.segment_name,
                "text_preview": cls.text_preview
            })
        
        # Group consecutive segment pages
        segment_groups = []
        current_group = None
        
        for cls in classifications:
            if cls.page_type == "segment":
                if current_group is None:
                    current_group = {
                        "segment_name": cls.segment_name,
                        "pages": [cls.page_number],
                        "start_page": cls.page_number
                    }
                elif cls.segment_name == current_group["segment_name"]:
                    # Same segment continues
                    current_group["pages"].append(cls.page_number)
                else:
                    # New segment
                    segment_groups.append(current_group)
                    current_group = {
                        "segment_name": cls.segment_name,
                        "pages": [cls.page_number],
                        "start_page": cls.page_number
                    }
            else:
                if current_group is not None:
                    segment_groups.append(current_group)
                    current_group = None
        
        if current_group is not None:
            segment_groups.append(current_group)
        
        # Group consecutive index pages
        index_groups = []
        current_index = None
        
        for cls in classifications:
            if cls.page_type == "index":
                if current_index is None:
                    current_index = {
                        "pages": [cls.page_number],
                        "start_page": cls.page_number
                    }
                elif cls.page_number == current_index["pages"][-1] + 1:
                    current_index["pages"].append(cls.page_number)
                else:
                    index_groups.append(current_index)
                    current_index = {
                        "pages": [cls.page_number],
                        "start_page": cls.page_number
                    }
            else:
                if current_index is not None:
                    index_groups.append(current_index)
                    current_index = None
        
        if current_index is not None:
            index_groups.append(current_index)
        
        return {
            "total_pages": len(classifications),
            "summary": {
                "index_pages": len(grouped["index"]),
                "segment_pages": len(grouped["segment"]),
                "other_pages": len(grouped["other"])
            },
            "index_groups": index_groups,
            "segment_groups": segment_groups,
            "other_pages": [p["page_number"] for p in grouped["other"]],
            "detailed_classifications": [
                {
                    "page_number": cls.page_number,
                    "type": cls.page_type,
                    "confidence": cls.confidence,
                    "segment_name": cls.segment_name
                }
                for cls in classifications
            ]
        }


def classify_pdf(
    pdf_path: str,
    model_path: str = "models/best_model.pt",
    tokenizer_path: str = "models/tokenizer",
    output_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to classify a PDF and optionally save results.
    """
    classifier = PageClassifierInference(
        model_path=model_path,
        tokenizer_path=tokenizer_path
    )
    
    results = classifier.classify_pdf_grouped(pdf_path)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify 850 Purchase Order PDF pages")
    parser.add_argument("pdf", type=str, help="Path to PDF file")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/tokenizer",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    results = classify_pdf(
        args.pdf,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        output_path=args.output
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\nTotal pages: {results['total_pages']}")
    print(f"Index pages: {results['summary']['index_pages']}")
    print(f"Segment pages: {results['summary']['segment_pages']}")
    print(f"Other pages: {results['summary']['other_pages']}")
    
    print("\n--- Index Page Groups ---")
    for group in results["index_groups"]:
        pages = ", ".join(str(p) for p in group["pages"])
        print(f"  Pages: {pages}")
    
    print("\n--- Segment Groups ---")
    for group in results["segment_groups"]:
        pages = ", ".join(str(p) for p in group["pages"])
        print(f"  {group['segment_name']}: Pages {pages}")
    
    print("\n--- Other Pages ---")
    print(f"  Pages: {', '.join(str(p) for p in results['other_pages'])}")
    
    print("\n--- Detailed Classifications ---")
    for cls in results["detailed_classifications"]:
        segment_info = f" ({cls['segment_name']})" if cls['segment_name'] else ""
        print(f"  Page {cls['page_number']}: {cls['type']}{segment_info} "
              f"(confidence: {cls['confidence']:.3f})")


if __name__ == "__main__":
    main()
