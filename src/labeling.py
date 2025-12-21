"""
Weak Supervision Labeling Module.
Generates pseudo-labels for training data based on heuristics.
No ground truth is used - only rule-based patterns.
"""

import re
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Label definitions
LABEL_INDEX = "index"
LABEL_SEGMENT = "segment"
LABEL_OTHER = "other"

LABELS = {LABEL_INDEX: 0, LABEL_SEGMENT: 1, LABEL_OTHER: 2}
LABEL_NAMES = {v: k for k, v in LABELS.items()}

# Known X12 EDI segment IDs
KNOWN_SEGMENTS = {
    "ISA", "GS", "ST", "SE", "GE", "IEA",
    "BEG", "CUR", "REF", "PER", "TAX", "FOB", "SAC", "ITD", "DTM",
    "TD5", "N9", "MSG", "N1", "N2", "N3", "N4", "PO1", "CTP", "PID",
    "SCH", "CTT", "AMT", "MAN", "PKG", "MEA", "PWK", "CSH", "TC2",
    "SLN", "TXI", "QTY", "SDQ", "LDT", "ACK", "SPI", "PAM", "G61"
}


@dataclass
class LabelResult:
    """Result of labeling a page."""
    label: str
    confidence: float
    segment_id: Optional[str] = None
    reason: str = ""


def normalize_text(text: str) -> str:
    """Normalize text for pattern matching."""
    return " ".join(text.lower().split())


def compute_index_score(text: str) -> Tuple[int, List[str]]:
    """
    Compute a score indicating likelihood this is an index/TOC page.
    Returns (score, list of matched patterns).
    """
    patterns = [
        (r"table\s+of\s+contents", 3),
        (r"segment\s+summary", 2),
        (r"pos\.?\s+seg\.?\s+id", 2),
        (r"page\s+pos\.?\s+seg", 2),
        (r"loop\s+id", 2),
        (r"functional\s+group", 1),
        (r"heading:", 1),
        (r"detail:", 1),
        (r"summary:", 1),
        (r"base\s+user", 1),
        (r"max\.?\s*use", 1),
        (r"req\.?\s+des", 1),
    ]
    
    score = 0
    matched = []
    
    for pattern, weight in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
            matched.append(pattern)
    
    # Check for tabular structure with segment listings
    # Pattern: multiple lines with "NNN SEG Description"
    segment_rows = re.findall(r"^\s*\d+\s+[A-Z]{2,3}\s+[A-Za-z]", text, re.MULTILINE)
    if len(segment_rows) >= 5:
        score += 3
        matched.append("segment_table_rows")
    
    # IMPROVED: Additional index patterns
    # Multiple segment IDs listed in table format
    seg_ids_in_table = re.findall(r"^\s*(?:\d+\s+)?([A-Z]{2,3})\s+\d", text, re.MULTILINE)
    known_segs = [s for s in seg_ids_in_table if s in KNOWN_SEGMENTS]
    if len(known_segs) >= 8:
        score += 3
        matched.append(f"segment_listing({len(known_segs)})")
    
    # "Segment Directory" or similar headers
    if re.search(r"segment\s+(directory|index|list)", text, re.IGNORECASE):
        score += 2
        matched.append("segment_directory")
    
    # Table header patterns common in index pages
    if re.search(r"(Seg|ID|Name|Req|Max|Loop)\s+(Seg|ID|Name|Req|Max|Loop)", text):
        score += 2
        matched.append("table_headers")
    
    # M/O (Mandatory/Optional) column in tables
    mo_patterns = re.findall(r"\b[MO]\b\s+\d+", text)
    if len(mo_patterns) >= 5:
        score += 2
        matched.append(f"mo_column({len(mo_patterns)})")
    
    return score, matched


def extract_segment_id(text: str) -> Optional[str]:
    """
    Extract segment ID from page text.
    Returns segment ID (e.g., 'ST', 'BEG') if found.
    """
    lines = text.split('\n')
    
    # Pattern 1: "Segment: ST" or "Segment: ST Transaction Set Header"
    match = re.search(r"Segment:\s*([A-Z][A-Z0-9]{0,2})\b", text)
    if match:
        seg_id = match.group(1).upper()
        if seg_id in KNOWN_SEGMENTS or len(seg_id) <= 3:
            return seg_id
    
    # Pattern 2: Segment ID at start of line followed by description
    # e.g., "ST  Transaction Set Header"
    for line in lines[:15]:
        clean = line.strip()
        if not clean:
            continue
        
        parts = clean.split()
        if not parts:
            continue
        
        first = parts[0].upper()
        
        # Must be 2-3 uppercase letters/digits, matching known segments
        if first in KNOWN_SEGMENTS:
            # Verify context - should have description or be followed by element info
            if len(parts) > 1 or re.search(r"(pos|element|loop|level)", text.lower()):
                return first
    
    # Pattern 3: Standalone segment header like "BEG\nBeginning Segment"
    for i, line in enumerate(lines[:10]):
        clean = line.strip().upper()
        if clean in KNOWN_SEGMENTS:
            # Check next line for description
            if i + 1 < len(lines) and lines[i + 1].strip():
                return clean
    
    return None


def compute_segment_score(text: str) -> Tuple[int, List[str]]:
    """
    Compute a score indicating likelihood this is a segment definition page.
    """
    patterns = [
        (r"Segment:\s*[A-Z]", 3),
        (r"Position:\s*\d", 2),
        (r"Loop:\s*\w", 2),
        (r"Level:\s*(Heading|Detail|Summary)", 2),
        (r"Usage:\s*(Mandatory|Optional|Conditional)", 2),
        (r"Max\s*Use:", 1),
        (r"Data\s+Element\s+Summary", 2),
        (r"Element\s+Summary", 2),
        (r"Ref\.?\s+Data", 1),
        (r"Pos:\s*\d", 1),
        (r"Max:\s*\d", 1),
        (r"(M|O|C)\s+(AN|ID|DT|TM|N\d|R)\s+\d+/\d+", 2),  # Element type patterns
    ]
    
    score = 0
    matched = []
    
    for pattern, weight in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
            matched.append(pattern)
    
    # Check if we found a segment ID
    if extract_segment_id(text):
        score += 2
        matched.append("segment_id_found")
    
    # IMPROVED: Check for segment continuation patterns
    # Element reference patterns like PO101, REF01, N101, ISA01
    element_refs = re.findall(r"\b[A-Z]{2,3}\d{2}\b", text)
    if len(element_refs) >= 3:
        score += 3
        matched.append(f"element_refs({len(element_refs)})")
    
    # Syntax Rules / Semantics sections (common in continuations)
    if re.search(r"Syntax\s*Rules?\s*:", text, re.IGNORECASE):
        score += 2
        matched.append("syntax_rules")
    
    if re.search(r"Semantics?\s*:", text, re.IGNORECASE):
        score += 2
        matched.append("semantics")
    
    # Code tables with "Code Name" pattern
    code_names = re.findall(r"^\s*[A-Z0-9]{1,3}\s+[A-Z][a-z]", text, re.MULTILINE)
    if len(code_names) >= 5:
        score += 2
        matched.append(f"code_table({len(code_names)})")
    
    # Element type definitions (AN, ID, DT, N0, R followed by length)
    elem_types = re.findall(r"\b(AN|ID|DT|TM|N\d|R)\s+\d+/\d+", text)
    if len(elem_types) >= 2:
        score += 2
        matched.append(f"elem_types({len(elem_types)})")
    
    # "Description:" pattern common in element definitions
    desc_count = len(re.findall(r"Description:", text, re.IGNORECASE))
    if desc_count >= 2:
        score += 1
        matched.append(f"descriptions({desc_count})")
    
    return score, matched


def compute_other_score(text: str) -> Tuple[int, List[str]]:
    """
    Compute a score indicating likelihood this is an "other" page.
    """
    patterns = [
        (r"sample\s+data", 3),
        (r"preface", 2),
        (r"introduction", 2),
        (r"trading\s+partner", 2),
        (r"purpose\s+and\s+scope", 2),
        (r"revision\s+history", 2),
        (r"contact\s+information", 1),
        (r"copyright", 1),
    ]
    
    score = 0
    matched = []
    
    for pattern, weight in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
            matched.append(pattern)
    
    # Very short pages are likely other
    word_count = len(text.split())
    if word_count < 30:
        score += 2
        matched.append("short_page")
    
    return score, matched


def label_page(text: str, prev_label: str = None, prev_segment: str = None) -> LabelResult:
    """
    Assign a label to a page based on heuristics.
    
    Args:
        text: Page text content
        prev_label: Label of previous page (for continuation detection)
        prev_segment: Segment ID of previous page
    
    Returns:
        LabelResult with label, confidence, and optional segment_id
    """
    # Compute scores for each class
    index_score, index_matches = compute_index_score(text)
    segment_score, segment_matches = compute_segment_score(text)
    other_score, other_matches = compute_other_score(text)
    
    # Extract segment ID if present
    segment_id = extract_segment_id(text)
    
    # IMPROVED: Compare relative scores instead of fixed thresholds
    
    # 1. Strong other indicators (short pages, preface, etc.)
    if other_score >= 3 and segment_score < 3 and index_score < 3:
        return LabelResult(
            label=LABEL_OTHER,
            confidence=min(0.90, 0.5 + other_score * 0.1),
            reason=f"other_patterns: {other_matches}"
        )
    
    # 2. Segment with explicit ID - highest priority for segment pages
    if segment_id and segment_score >= 2:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=min(0.95, 0.7 + segment_score * 0.02),
            segment_id=segment_id,
            reason=f"segment_id={segment_id}, patterns: {segment_matches}"
        )
    
    # 3. Compare segment vs index scores when both are significant
    if segment_score >= 4 and segment_score > index_score:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=min(0.90, 0.6 + segment_score * 0.02),
            segment_id=segment_id,
            reason=f"segment_patterns: {segment_matches}"
        )
    
    # 4. Strong index indicators (only if segment score is lower)
    if index_score >= 4 and index_score > segment_score:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=min(0.95, 0.6 + index_score * 0.05),
            reason=f"index_patterns: {index_matches}"
        )
    
    # 5. Segment based on strong structure patterns
    if segment_score >= 6:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=min(0.90, 0.6 + segment_score * 0.02),
            segment_id=segment_id,
            reason=f"segment_structure: {segment_matches}"
        )
    
    # 6. Continuation of previous segment
    if prev_label == LABEL_SEGMENT and segment_score >= 2:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=0.80,
            segment_id=prev_segment,
            reason="segment_continuation"
        )
    
    # 7. Moderate index score (only if clearly dominant)
    if index_score >= 3 and segment_score < 2:
        return LabelResult(
            label=LABEL_INDEX,
            confidence=0.70,
            reason=f"moderate_index: {index_matches}"
        )
    
    # 8. Weak segment indicators
    if segment_score >= 3:
        return LabelResult(
            label=LABEL_SEGMENT,
            confidence=0.65,
            segment_id=segment_id,
            reason=f"weak_segment: {segment_matches}"
        )
    
    # 9. Default to other
    return LabelResult(
        label=LABEL_OTHER,
        confidence=0.50,
        reason="default"
    )
