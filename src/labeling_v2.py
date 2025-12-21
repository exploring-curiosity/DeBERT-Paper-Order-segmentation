"""
Enhanced labeling system using font features and sequence context.
Designed to achieve 99%+ accuracy on page classification.
"""

import fitz
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from .features import (
    PageFeatures, extract_page_features, extract_document_features,
    KNOWN_SEGMENTS, detect_element_table
)


LABEL_INDEX = "index"
LABEL_SEGMENT = "segment"
LABEL_OTHER = "other"

LABELS = {LABEL_INDEX: 0, LABEL_SEGMENT: 1, LABEL_OTHER: 2}
LABEL_NAMES = {v: k for k, v in LABELS.items()}


@dataclass
class LabelResult:
    """Result of labeling a page."""
    label: str
    confidence: float
    segment_id: Optional[str] = None
    reason: str = ""
    is_continuation: bool = False


def classify_page_with_context(
    features: PageFeatures,
    prev_result: Optional[LabelResult] = None,
    next_features: Optional[PageFeatures] = None
) -> LabelResult:
    """
    Classify a page using features and sequence context.
    
    Key insights from error analysis:
    1. Continuation pages have smaller fonts and no segment headers
    2. Index second pages have "Summary:", "Notes:" headers
    3. "Other" pages include TOC and loop overview pages
    4. Segment first pages have segment ID in large font
    """
    
    # Count segment IDs in text for index detection
    seg_ids_found = set()
    for seg in KNOWN_SEGMENTS:
        if re.search(rf"\b{seg}\b", features.text):
            seg_ids_found.add(seg)
    
    # Rule 0: Glossary/Document Control pages → OTHER
    if features.has_glossary:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.95,
            reason="glossary_pattern"
        )
    
    # Rule 0b: Preface/Introduction pages → OTHER
    if features.has_preface:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.95,
            reason="preface_pattern"
        )
    
    # Rule 0c: Appendix/Example pages → OTHER
    if features.has_appendix:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.95,
            reason="appendix_pattern"
        )
    
    # Rule 1: Table of Contents → OTHER (but not if it's an index with segment list)
    if features.has_toc_pattern and len(seg_ids_found) < 5:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.98,
            reason="toc_pattern"
        )
    
    # Rule 1b: Index header patterns → INDEX
    if features.has_index_header:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.92,
            reason="index_header"
        )
    
    # Rule 2: Loop overview pages → OTHER (but be careful not to catch index pages)
    # Only if it doesn't have many segment IDs (which would indicate index)
    if features.has_loop_overview and not features.has_element_table and len(seg_ids_found) < 5:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.95,
            reason="loop_overview"
        )
    
    # Rule 3: Summary/Notes header (index continuation) → INDEX
    if features.has_summary_notes:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.95,
            reason="summary_notes_header"
        )
    
    # Rule 3b: Index page with many segment IDs in table format
    # Must have BOTH table patterns AND many segment IDs
    index_patterns = [
        r"Pos\.?\s+Seg",
        r"Seg\.?\s*ID\s+Name",
        r"Loop\s+ID",
        r"Req\.?\s+Des",
        r"Max\.?\s+Use",
        r"Pos\s+Id\s+Segment\s+Name",
        r"ID\s+Segment\s+Name\s+Req",
        r"Transactions?\s+Summary",
        r"Transaction\s+Set\s+Notes",
    ]
    has_index_pattern = any(re.search(p, features.text, re.IGNORECASE) for p in index_patterns)
    
    # Second index pages often have fewer segment IDs but still have table patterns
    # Lower threshold if we have strong index patterns
    min_seg_ids = 5 if has_index_pattern else 10
    
    # Classify as index if we have segment IDs AND index patterns
    if len(seg_ids_found) >= min_seg_ids and has_index_pattern:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.92,
            reason=f"segment_table({len(seg_ids_found)})"
        )
    
    # Rule 4: Segment header with ID → SEGMENT (first page)
    # But NOT if this looks like an index page (has index patterns)
    # Also check for "Transaction Set Notes" which is index, not segment
    if features.has_segment_header and features.segment_id and len(seg_ids_found) < 6 and not has_index_pattern:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=0.98,
            segment_id=features.segment_id,
            reason="segment_header",
            is_continuation=False
        )
    
    # Rule 5: Element table without segment header → check context
    if features.has_element_table:
        if prev_result and prev_result.label == LABEL_SEGMENT:
            # Continuation of previous segment
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.95,
                segment_id=prev_result.segment_id,
                reason="element_table_continuation",
                is_continuation=True
            )
        else:
            # First page of segment (header might not be detected)
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.90,
                reason="element_table_no_header"
            )
    
    # Rule 6: Strong segment patterns in text
    segment_patterns = [
        r"Pos:\s*\d+",
        r"Max:\s*\d+",
        r"Loop:\s*[A-Z/]+",
        r"Usage:\s*(Mandatory|Optional|Must use|Used)",
        r"\b[A-Z]{2,3}\d{2}\b",  # Element refs like PO101
    ]
    
    pattern_matches = sum(1 for p in segment_patterns if re.search(p, features.text, re.IGNORECASE))
    
    if pattern_matches >= 3:
        if prev_result and prev_result.label == LABEL_SEGMENT:
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.90,
                segment_id=prev_result.segment_id,
                reason=f"segment_patterns({pattern_matches})_continuation",
                is_continuation=True
            )
        else:
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.85,
                reason=f"segment_patterns({pattern_matches})"
            )
    
    # Rule 7: Context-based continuation
    # If previous page was segment and current has similar content patterns
    if prev_result and prev_result.label == LABEL_SEGMENT:
        # Check for element reference continuation
        elem_refs = re.findall(r"\b[A-Z]{2,3}\d{2}\b", features.text)
        if len(elem_refs) >= 2:
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.88,
                segment_id=prev_result.segment_id,
                reason="element_refs_continuation",
                is_continuation=True
            )
        
        # Check for syntax/semantics sections
        if re.search(r"Syntax\s*Rules?:|Semantics?:", features.text, re.IGNORECASE):
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.88,
                segment_id=prev_result.segment_id,
                reason="syntax_semantics_continuation",
                is_continuation=True
            )
        
        # Check for code table continuation
        code_lines = re.findall(r"^\s*[A-Z0-9]{1,3}\s+[A-Z][a-z]", features.text, re.MULTILINE)
        if len(code_lines) >= 5:
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.85,
                segment_id=prev_result.segment_id,
                reason="code_table_continuation",
                is_continuation=True
            )
        
        # Check for "Description:" pattern
        if features.text.count("Description:") >= 2:
            return LabelResult(
                label=LABEL_SEGMENT,
                confidence=0.82,
                segment_id=prev_result.segment_id,
                reason="description_continuation",
                is_continuation=True
            )
    
    # Rule 8: Index patterns (segment table)
    index_patterns = [
        r"Pos\.?\s+Seg\.?\s*ID",
        r"Req\.?\s+Des",
        r"Max\.?\s+Use",
        r"Loop\s+ID\s+Name\s+Req",
    ]
    
    index_matches = sum(1 for p in index_patterns if re.search(p, features.text, re.IGNORECASE))
    
    # Multiple segment IDs in tabular format
    seg_ids_found = set()
    for seg in KNOWN_SEGMENTS:
        if re.search(rf"\b{seg}\b", features.text):
            seg_ids_found.add(seg)
    
    if index_matches >= 1 and len(seg_ids_found) >= 5:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.90,
            reason=f"index_patterns({index_matches})_segments({len(seg_ids_found)})"
        )
    
    # Rule 9: Previous was index and current has similar patterns
    if prev_result and prev_result.label == LABEL_INDEX:
        if len(seg_ids_found) >= 3 and features.num_tables > 0:
            return LabelResult(
                label=LABEL_INDEX,
                confidence=0.85,
                reason="index_continuation"
            )
    
    # Rule 10: Very short/sparse pages → OTHER
    # But only if they don't have segment patterns
    text_len = len(features.text.strip())
    if text_len < 100:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.75,
            reason="very_short_page"
        )
    
    # Rule 11: Default based on scores
    if features.segment_score > features.index_score and features.segment_score > features.other_score:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=0.60,
            reason="score_based"
        )
    elif features.index_score > features.other_score:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.60,
            reason="score_based"
        )
    else:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=0.50,
            reason="default"
        )


def label_document(pdf_path: str) -> List[LabelResult]:
    """
    Label all pages in a document using sequence context.
    
    Key insights:
    1. "Other" pages mostly occur at START and END of documents
    2. Index pages list segments in order matching upcoming segment pages
    3. Once segments start, they continue until near the end
    """
    features_list = extract_document_features(pdf_path)
    total_pages = len(features_list)
    results = []
    
    # First pass: initial classification
    for i, features in enumerate(features_list):
        prev_result = results[-1] if results else None
        next_features = features_list[i + 1] if i + 1 < len(features_list) else None
        
        result = classify_page_with_context(features, prev_result, next_features)
        results.append(result)
    
    # Second pass: apply positional heuristics
    # Find first index page
    first_index_idx = None
    for i, r in enumerate(results):
        if r.label == LABEL_INDEX:
            first_index_idx = i
            break
    
    # Find first segment page
    first_segment_idx = None
    for i, r in enumerate(results):
        if r.label == LABEL_SEGMENT:
            first_segment_idx = i
            break
    
    # Apply positional corrections
    for i, (features, result) in enumerate(zip(features_list, results)):
        page_num = i + 1
        
        # Heuristic: Pages BEFORE the first index/segment are likely "other"
        if first_index_idx is not None and i < first_index_idx:
            if result.label != LABEL_OTHER and result.confidence < 0.9:
                # Check if it has strong segment indicators
                if not features.has_segment_header and not features.has_element_table:
                    results[i] = LabelResult(
                        label=LABEL_OTHER,
                        confidence=0.85,
                        reason="pre_index_position"
                    )
        
        # Heuristic: Last few pages are often "other" (appendix, examples)
        if page_num >= total_pages - 1:  # Last 2 pages
            if result.label == LABEL_SEGMENT and result.confidence < 0.85:
                if not features.has_segment_header:
                    results[i] = LabelResult(
                        label=LABEL_OTHER,
                        confidence=0.75,
                        reason="end_position"
                    )
        
        # Heuristic: Middle pages (between first segment and near-end) are rarely "other"
        if first_segment_idx is not None:
            if i > first_segment_idx and page_num < total_pages - 2:
                if result.label == LABEL_OTHER and result.reason == "default":
                    # Check if previous was segment - likely continuation
                    if i > 0 and results[i-1].label == LABEL_SEGMENT:
                        results[i] = LabelResult(
                            label=LABEL_SEGMENT,
                            confidence=0.80,
                            segment_id=results[i-1].segment_id,
                            reason="middle_position_continuation",
                            is_continuation=True
                        )
    
    return results


def generate_training_data(pdf_dir: str, output_path: str) -> List[dict]:
    """Generate training data from a directory of PDFs."""
    import json
    from tqdm import tqdm
    
    pdf_dir = Path(pdf_dir)
    all_data = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_path in tqdm(pdf_files, desc="Labeling PDFs"):
        try:
            doc = fitz.open(str(pdf_path))
            features_list = extract_document_features(str(pdf_path))
            results = label_document(str(pdf_path))
            
            for i, (features, result) in enumerate(zip(features_list, results)):
                all_data.append({
                    "filename": pdf_path.name,
                    "page_number": i + 1,
                    "text": features.text,
                    "label": result.label,
                    "confidence": result.confidence,
                    "segment_id": result.segment_id,
                    "is_continuation": result.is_continuation,
                    "reason": result.reason,
                    # Additional features for training
                    "max_font_size": features.max_font_size,
                    "has_segment_header": features.has_segment_header,
                    "has_toc_pattern": features.has_toc_pattern,
                    "has_element_table": features.has_element_table,
                })
            
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    # Summary
    label_counts = {}
    for item in all_data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nLabeling summary ({len(all_data)} pages):")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(all_data)*100:.1f}%)")
    
    avg_conf = sum(item["confidence"] for item in all_data) / len(all_data)
    print(f"Average confidence: {avg_conf:.3f}")
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    return all_data
