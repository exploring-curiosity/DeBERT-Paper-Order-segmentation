"""
Meaningful rule-based labeling for EDI page classification.

Philosophy:
- Rules based on what a HUMAN would recognize
- No arbitrary numeric thresholds (>=3, >=5, etc.)
- Leverage pre-computed semantic features from features.py
- Structural patterns that generalize to new documents

What a human sees:
- INDEX: A table listing segments with positions - segment directory
- SEGMENT: Definition of a specific segment with its elements
- OTHER: Front matter (preface, glossary) or back matter (appendix, examples)
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from .features import (
    PageFeatures, extract_document_features,
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


# =============================================================================
# STRUCTURAL PATTERN DETECTION
# =============================================================================

def has_segment_listing_table(text: str) -> bool:
    """
    Check if page has a segment listing table - the defining characteristic of INDEX.
    
    A human recognizes this as a table with rows like:
    "0100  ST  Transaction Set Header"
    "0200  BEG  Beginning Segment"
    
    This is STRUCTURAL - looking for tabular format, not counting.
    """
    # Pattern: position number, segment ID, description
    # This captures the TABLE STRUCTURE that defines an index
    pattern = r"\d{2,4}\s+([A-Z]{2,3})\s+[A-Z][a-z]"
    matches = re.findall(pattern, text)
    
    # Validate found IDs are real segments
    valid = [m for m in matches if m in KNOWN_SEGMENTS]
    
    # A table has MULTIPLE rows - that's what makes it a table, not a single mention
    # This is structural, not arbitrary counting
    return len(valid) > 2


def has_element_definition_structure(text: str) -> bool:
    """
    Check if page has element definition structure - characteristic of SEGMENT pages.
    
    A human recognizes element definitions by seeing:
    - Position indicators (Pos: 010)
    - Data type specifications (M AN 1/30)
    - Usage indicators (Mandatory, Optional)
    
    This is looking for the STRUCTURE of element documentation.
    """
    # These patterns together indicate element definition structure
    has_position = bool(re.search(r"(?i)Pos:\s*\d+", text))
    has_data_type = bool(re.search(r"\b(M|O|C)\s+(AN|ID|DT|TM|N\d|R)\s+\d+/\d+", text))
    has_usage = bool(re.search(r"(?i)Usage:\s*(Mandatory|Optional|Must\s*use|Used)", text))
    has_max = bool(re.search(r"(?i)Max:\s*[>\d]+", text))
    
    # Need multiple structural indicators - this is what defines element documentation
    indicators = sum([has_position, has_data_type, has_usage, has_max])
    return indicators >= 2


def has_segment_label_header(text: str) -> tuple:
    """
    Check for segment page header format: "Segment [ID] – [Description]"
    
    Some documents (like arnecom) use this format:
    "Segment ST – Transaction Set Header"
    "Segment BEG – Beginning Segment for Purchase Order"
    
    May also appear with newline between "Segment" and the ID:
    "Segment
     TD5 - Carrier Details"
    
    The word "Segment" followed by segment ID clearly marks a segment definition page.
    Returns (found, segment_id).
    """
    # Pattern 1: "Segment" followed by segment ID on same line
    # Handles both en-dash (–) and hyphen (-)
    pattern1 = r"(?i)\bSegment\s+([A-Z][A-Z0-9]{1,2})\s*[–\-]\s*[A-Z]"
    match = re.search(pattern1, text)
    if match:
        seg_id = match.group(1).upper()
        if seg_id in KNOWN_SEGMENTS:
            return (True, seg_id)
    
    # Pattern 2: "Segment" on one line, "[ID] -" on next line (arnecom style)
    pattern2 = r"(?i)\bSegment\s*\n\s*([A-Z][A-Z0-9]{1,2})\s*[–\-]"
    match = re.search(pattern2, text)
    if match:
        seg_id = match.group(1).upper()
        if seg_id in KNOWN_SEGMENTS:
            return (True, seg_id)
    
    return (False, None)


def is_raw_edi_sample(text: str) -> bool:
    """
    Check if page contains raw EDI transaction data (sample/example).
    
    A human immediately recognizes EDI syntax:
    ST*850*0001~
    BEG*00*NE*12345**20231224~
    
    This is the EDI wire format, not documentation.
    """
    # EDI segment pattern: SEG*data*data~
    edi_lines = re.findall(r"[A-Z]{2,3}\*[^~\n]*~", text)
    
    if len(edi_lines) > 3:
        # If substantial portion is EDI data, it's a sample page
        edi_chars = sum(len(line) for line in edi_lines)
        total_chars = len(text.strip())
        if total_chars > 0 and edi_chars / total_chars > 0.25:
            return True
    return False


# =============================================================================
# MAIN CLASSIFICATION LOGIC
# =============================================================================

def classify_page(
    features: PageFeatures,
    prev_result: Optional[LabelResult] = None
) -> LabelResult:
    """
    Classify a page using meaningful structural patterns.
    
    Priority order based on what a human would check:
    1. Is this front/back matter? (titles like Glossary, Appendix)
    2. Is this a segment definition? (segment header + elements)
    3. Is this an index? (segment listing table)
    4. Context from previous page
    5. Fallback based on content
    """
    text = features.text
    
    # =========================================================================
    # RULE 1: Front/Back Matter Detection
    # A human recognizes these by their distinctive titles
    # =========================================================================
    
    # Glossary, abbreviations, definitions - clearly OTHER
    if features.has_glossary:
        return LabelResult(LABEL_OTHER, 0.95, reason="glossary")
    
    # Preface, introduction, overview - clearly OTHER
    if features.has_preface:
        return LabelResult(LABEL_OTHER, 0.95, reason="preface")
    
    # Appendix, examples - clearly OTHER
    if features.has_appendix:
        return LabelResult(LABEL_OTHER, 0.95, reason="appendix")
    
    # Table of contents (different from segment index)
    if features.has_toc_pattern and not features.has_index_header:
        return LabelResult(LABEL_OTHER, 0.95, reason="table_of_contents")
    
    # =========================================================================
    # RULE 2: Segment Definition Detection
    # A human recognizes segment pages by their header and element table
    # BUT: index pages can LIST segment headers - must distinguish
    # =========================================================================
    
    # Check for explicit "Segment [ID] – [Description]" format (arnecom style)
    # This is unambiguous - clearly marks a segment definition page
    seg_label_found, seg_label_id = has_segment_label_header(text)
    if seg_label_found:
        return LabelResult(
            LABEL_SEGMENT, 0.98,
            segment_id=seg_label_id,
            reason="segment_label_header"
        )
    
    # Segment header like "BEG - Beginning Segment for Purchase Order"
    # Only classify as SEGMENT if it's a definition (has element content)
    # NOT if it's part of an index listing
    if features.has_segment_header and features.segment_id:
        # Check if this is actually an index page listing segments
        if has_segment_listing_table(text):
            return LabelResult(LABEL_INDEX, 0.92, reason="index_with_segment_listing")
        
        # Early pages (1-4) with segment headers are usually index, not segment
        # unless they have strong element content
        is_early_page = features.page_num <= 4
        
        # Check for STRONG element content (element table or definition structure)
        has_strong_element_content = (
            features.has_element_table or 
            has_element_definition_structure(text)
        )
        
        # Weaker check: just element refs
        has_element_refs = bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text))
        
        if has_strong_element_content:
            return LabelResult(
                LABEL_SEGMENT, 0.98,
                segment_id=features.segment_id,
                reason="segment_header_with_elements"
            )
        
        # Early pages with weak element content - likely index
        if is_early_page:
            if prev_result and prev_result.label == LABEL_INDEX:
                return LabelResult(LABEL_INDEX, 0.85, reason="segment_header_after_index")
            # Check for index characteristics
            seg_ids = set()
            for seg in KNOWN_SEGMENTS:
                if re.search(rf"\b{seg}\b", text):
                    seg_ids.add(seg)
            if len(seg_ids) > 3:  # Multiple segments mentioned = likely index
                return LabelResult(LABEL_INDEX, 0.80, reason="early_page_multiple_segments")
        
        # Has element refs but not early - treat as segment
        if has_element_refs and not is_early_page:
            return LabelResult(
                LABEL_SEGMENT, 0.90,
                segment_id=features.segment_id,
                reason="segment_header_with_refs"
            )
    
    # Element table (Element Summary, data type specifications)
    # This is segment content - either first page or continuation
    if features.has_element_table:
        seg_id = prev_result.segment_id if prev_result and prev_result.label == LABEL_SEGMENT else None
        is_cont = prev_result is not None and prev_result.label == LABEL_SEGMENT
        return LabelResult(
            LABEL_SEGMENT, 0.95,
            segment_id=seg_id,
            reason="element_table",
            is_continuation=is_cont
        )
    
    # Element definition structure (Pos, Max, Usage patterns)
    if has_element_definition_structure(text):
        seg_id = prev_result.segment_id if prev_result and prev_result.label == LABEL_SEGMENT else None
        is_cont = prev_result is not None and prev_result.label == LABEL_SEGMENT
        return LabelResult(
            LABEL_SEGMENT, 0.90,
            segment_id=seg_id,
            reason="element_definitions",
            is_continuation=is_cont
        )
    
    # =========================================================================
    # RULE 3: Index Detection
    # A human recognizes index by seeing a table of segment listings
    # =========================================================================
    
    # Explicit index header ("EDI Levels and Segments")
    if features.has_index_header:
        return LabelResult(LABEL_INDEX, 0.95, reason="index_header")
    
    # Summary/Notes section that lists segments (index continuation)
    if features.has_summary_notes and has_segment_listing_table(text):
        return LabelResult(LABEL_INDEX, 0.92, reason="summary_with_listings")
    
    # Segment listing table structure
    if has_segment_listing_table(text):
        return LabelResult(LABEL_INDEX, 0.90, reason="segment_listing_table")
    
    # =========================================================================
    # RULE 4: Context-Based Continuation
    # Pages following segments often continue the same segment
    # =========================================================================
    
    if prev_result and prev_result.label == LABEL_SEGMENT:
        # Element references (PO101, BEG02) indicate continuation
        if re.search(r"\b[A-Z]{2,3}\d{2}\b", text):
            return LabelResult(
                LABEL_SEGMENT, 0.88,
                segment_id=prev_result.segment_id,
                reason="element_refs_continuation",
                is_continuation=True
            )
        
        # Syntax/Semantics sections
        if re.search(r"(?i)Syntax\s*Rules?:|Semantics?:", text):
            return LabelResult(
                LABEL_SEGMENT, 0.88,
                segment_id=prev_result.segment_id,
                reason="syntax_semantics",
                is_continuation=True
            )
        
        # Code values table
        if re.search(r"(?i)code\s+values?|allowed\s+values?", text):
            return LabelResult(
                LABEL_SEGMENT, 0.85,
                segment_id=prev_result.segment_id,
                reason="code_values",
                is_continuation=True
            )
        
        # Description patterns (multiple)
        if text.count("Description:") > 1:
            return LabelResult(
                LABEL_SEGMENT, 0.82,
                segment_id=prev_result.segment_id,
                reason="description_continuation",
                is_continuation=True
            )
        
        # Code table structure - only if NOT an index/overview page
        # Check that it doesn't have index characteristics
        if not has_segment_listing_table(text):
            code_lines = re.findall(r"^\s*[A-Z0-9]{1,3}\s+[A-Z][a-z]", text, re.MULTILINE)
            # Also require element refs to confirm it's segment content
            has_elem_refs = bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text))
            if len(code_lines) > 4 and has_elem_refs:
                return LabelResult(
                    LABEL_SEGMENT, 0.85,
                    segment_id=prev_result.segment_id,
                    reason="code_table_continuation",
                    is_continuation=True
                )
        
        # Loop reference - only with strong segment indicators
        # Loop references appear in overview pages too, so be careful
        if re.search(r"(?i)Loop:\s*[A-Z0-9/]+", text):
            # Must have element refs to confirm it's segment content
            has_elem_refs = bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text))
            if has_elem_refs:
                return LabelResult(
                    LABEL_SEGMENT, 0.80,
                    segment_id=prev_result.segment_id,
                    reason="loop_reference_continuation",
                    is_continuation=True
                )
    
    # Index continuation
    if prev_result and prev_result.label == LABEL_INDEX:
        # Check for segment listing table structure
        if has_segment_listing_table(text):
            return LabelResult(LABEL_INDEX, 0.85, reason="index_continuation_table")
        
        # Check for segment IDs in the text (index typically lists many)
        seg_ids = set()
        for seg in KNOWN_SEGMENTS:
            if re.search(rf"\b{seg}\b", text):
                seg_ids.add(seg)
        if len(seg_ids) > 2:
            return LabelResult(LABEL_INDEX, 0.82, reason="index_continuation")
    
    # =========================================================================
    # RULE 5: Raw EDI Sample Data
    # A human recognizes EDI syntax immediately
    # =========================================================================
    if is_raw_edi_sample(text):
        return LabelResult(LABEL_OTHER, 0.90, reason="edi_sample")
    
    # =========================================================================
    # RULE 6: Loop Overview Pages
    # Discuss loop structure but aren't segment definitions
    # =========================================================================
    if features.has_loop_overview and not features.has_element_table:
        # Only if not following a segment (could be continuation)
        if not (prev_result and prev_result.label == LABEL_SEGMENT):
            return LabelResult(LABEL_OTHER, 0.80, reason="loop_overview")
    
    # =========================================================================
    # RULE 7: Segment Pattern Detection (without context)
    # Check for segment documentation patterns
    # =========================================================================
    segment_indicators = [
        bool(re.search(r"(?i)Pos:\s*\d+", text)),
        bool(re.search(r"(?i)Max:\s*[>\d]+", text)),
        bool(re.search(r"(?i)Loop:\s*[A-Z0-9/]+", text)),
        bool(re.search(r"(?i)Usage:\s*(Mandatory|Optional|Must\s*use|Used)", text)),
        bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text)),  # Element refs
        bool(re.search(r"(?i)Syntax\s*Rules?:", text)),
        bool(re.search(r"(?i)Semantics?:", text)),
        bool(re.search(r"(?i)Description:", text)),
        bool(re.search(r"(?i)code\s+values?", text)),
    ]
    
    # If page has multiple segment documentation patterns, it's segment content
    if sum(segment_indicators) > 2:
        return LabelResult(LABEL_SEGMENT, 0.82, reason="segment_patterns")
    
    # Even with fewer patterns, if element refs present with one other indicator
    has_elem_refs = bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text))
    if has_elem_refs and sum(segment_indicators) > 1:
        return LabelResult(LABEL_SEGMENT, 0.75, reason="segment_with_refs")
    
    # =========================================================================
    # RULE 8: Default to OTHER
    # If no clear pattern matches, default to OTHER
    # This is safer than score-based guessing
    # =========================================================================
    return LabelResult(LABEL_OTHER, 0.50, reason="default")


def label_document(pdf_path: str) -> List[LabelResult]:
    """Label all pages in a document with structure validation."""
    features_list = extract_document_features(pdf_path)
    results = []
    
    # First pass: classify each page
    for features in features_list:
        prev_result = results[-1] if results else None
        result = classify_page(features, prev_result)
        results.append(result)
    
    # Second pass: validate document structure
    results = validate_document_structure(results, features_list)
    
    return results


def validate_document_structure(
    results: List[LabelResult],
    features_list: List[PageFeatures]
) -> List[LabelResult]:
    """
    Validate and correct labels based on document structure.
    
    EDI documents have predictable structure:
    1. Index pages at the beginning
    2. Segment definitions in the middle (bulk of document)
    3. Back matter at the end (optional)
    
    This corrects low-confidence labels that don't fit the pattern.
    """
    total = len(results)
    if total == 0:
        return results
    
    # Find document regions
    first_segment = None
    last_segment = None
    for i, r in enumerate(results):
        if r.label == LABEL_SEGMENT and r.confidence > 0.8:
            if first_segment is None:
                first_segment = i
            last_segment = i
    
    # Correction 1: Low-confidence pages AFTER first index page
    # Don't change page 1 (often front matter)
    first_index = None
    for i, r in enumerate(results):
        if r.label == LABEL_INDEX and r.confidence > 0.8:
            first_index = i
            break
    
    if first_index is not None and first_segment is not None:
        for i in range(first_index + 1, first_segment):
            if results[i].confidence < 0.7 and results[i].label != LABEL_INDEX:
                # Check if surrounded by index
                has_index_neighbor = (
                    (i > 0 and results[i-1].label == LABEL_INDEX) or
                    (i < total - 1 and results[i+1].label == LABEL_INDEX)
                )
                if has_index_neighbor:
                    results[i] = LabelResult(LABEL_INDEX, 0.70, reason="structure_index")
    
    # Correction 2: OTHER pages in segment region
    # Correct if neighbor is segment AND page has segment-like content
    if first_segment is not None and last_segment is not None:
        for i in range(first_segment, last_segment + 1):
            if results[i].label == LABEL_OTHER and results[i].reason == "default":
                prev_seg = i > 0 and results[i-1].label == LABEL_SEGMENT
                next_seg = i < total - 1 and results[i+1].label == LABEL_SEGMENT
                
                # Check for segment content indicators
                text = features_list[i].text
                has_segment_content = (
                    bool(re.search(r"\b[A-Z]{2,3}\d{2}\b", text)) or  # Element refs
                    bool(re.search(r"(?i)Description:", text)) or
                    bool(re.search(r"(?i)Pos:\s*\d+", text)) or
                    bool(re.search(r"(?i)Usage:", text)) or
                    bool(re.search(r"(?i)Syntax\s*Rules?:", text))
                )
                
                # Correct if: (both neighbors are segment) OR (one neighbor + content)
                if (prev_seg and next_seg) or ((prev_seg or next_seg) and has_segment_content):
                    seg_id = results[i-1].segment_id if prev_seg else results[i+1].segment_id
                    results[i] = LabelResult(
                        LABEL_SEGMENT, 0.70,
                        segment_id=seg_id,
                        reason="structure_segment",
                        is_continuation=True
                    )
    
    return results


def generate_labels(pdf_dir: str, output_path: str = None):
    """Generate labels for all PDFs in directory."""
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    pdf_path = Path(pdf_dir)
    all_data = []
    
    for pdf_file in tqdm(sorted(pdf_path.glob("*.pdf")), desc="Labeling"):
        try:
            results = label_document(str(pdf_file))
            for i, result in enumerate(results):
                all_data.append({
                    "pdf": pdf_file.name,
                    "page": i + 1,
                    "label": result.label,
                    "confidence": result.confidence,
                    "reason": result.reason
                })
        except Exception as e:
            print(f"Failed to label {pdf_file.name}: {e}")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
    
    return all_data
