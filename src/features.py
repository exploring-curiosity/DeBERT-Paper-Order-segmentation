"""
Enhanced feature extraction for page classification.
Extracts font sizes, layout features, and structural patterns.
"""

import fitz
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


# Known X12 segments
KNOWN_SEGMENTS = {
    "ISA", "GS", "ST", "BEG", "CUR", "REF", "PER", "TAX", "FOB", "ITD",
    "DTM", "DIS", "INC", "LIN", "SI", "PID", "MEA", "PWK", "PKG", "TD1",
    "TD5", "TD3", "TD4", "MAN", "PCT", "CTB", "TXI", "SAC", "CTP", "PAM",
    "N9", "N1", "N2", "N3", "N4", "PO1", "PO3", "PO4", "CTT", "AMT",
    "SE", "GE", "IEA", "CSH", "TC2", "SLN", "SDQ", "SCH", "MSG", "SPO", "ADV"
}


@dataclass
class PageFeatures:
    """Enhanced features for a single page."""
    page_num: int
    text: str
    
    # Font features
    max_font_size: float
    min_font_size: float
    avg_font_size: float
    font_size_std: float
    header_font_size: float  # Font size of first significant text
    
    # Content features
    has_segment_header: bool
    segment_id: Optional[str]
    has_toc_pattern: bool
    has_loop_overview: bool
    has_summary_notes: bool
    has_element_table: bool
    has_glossary: bool  # Glossary/abbreviation page
    has_index_header: bool  # "EDI levels and Segments" etc.
    has_preface: bool  # Preface/introduction page
    has_appendix: bool  # Appendix/example page
    
    # Layout features
    num_tables: int
    text_density: float
    num_text_blocks: int
    
    # Derived scores
    index_score: float
    segment_score: float
    other_score: float


def extract_font_features(page) -> Tuple[float, float, float, float, float, List[dict]]:
    """Extract font size statistics from a page."""
    fonts = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    first_significant_size = 0
    
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text and len(text) > 2:
                        size = span["size"]
                        fonts.append({"size": size, "text": text})
                        if first_significant_size == 0 and len(text) > 5:
                            first_significant_size = size
    
    if not fonts:
        return 0, 0, 0, 0, 0, []
    
    sizes = [f["size"] for f in fonts]
    max_size = max(sizes)
    min_size = min(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    # Standard deviation
    variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
    std_size = variance ** 0.5
    
    return max_size, min_size, avg_size, std_size, first_significant_size, fonts


def detect_segment_header(fonts: List[dict], text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if page has a segment header.
    Segment headers are typically in larger font near the top.
    """
    if not fonts:
        return False, None
    
    # Get max font size
    max_size = max(f["size"] for f in fonts)
    avg_size = sum(f["size"] for f in fonts) / len(fonts)
    
    # Check top fonts for segment ID
    for f in fonts[:30]:
        text_clean = f["text"].strip()
        
        # Must be in larger font
        if f["size"] < avg_size * 1.05:
            continue
        
        # Check for exact segment ID match
        for seg in KNOWN_SEGMENTS:
            if text_clean == seg:
                return True, seg
            if text_clean.startswith(seg + " ") or text_clean.startswith(seg + "\n"):
                return True, seg
        
        # Check "Segment: XXX" pattern
        match = re.search(r"^Segment:\s*([A-Z]{2,3})\b", text_clean)
        if match and match.group(1) in KNOWN_SEGMENTS:
            return True, match.group(1)
    
    # Check raw text for segment header patterns
    lines = text.split('\n')[:30]
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Pattern: "XXX - Segment Name" or "XXX Segment Name"
        match = re.match(r"^([A-Z]{2,3})\s*[-–]\s*[A-Z]", line)
        if match and match.group(1) in KNOWN_SEGMENTS:
            return True, match.group(1)
        
        # Pattern: "Segment XXX - Name" or "Segment XXX – Name"
        match = re.match(r"^Segment\s+([A-Z]{2,3})\s*[-–]", line, re.IGNORECASE)
        if match and match.group(1) in KNOWN_SEGMENTS:
            return True, match.group(1)
        
        # Pattern: "Segment: XXX"
        match = re.match(r"^Segment:\s*([A-Z]{2,3})\b", line, re.IGNORECASE)
        if match and match.group(1) in KNOWN_SEGMENTS:
            return True, match.group(1)
        
        # Pattern: standalone segment ID on a line
        if line in KNOWN_SEGMENTS:
            return True, line
        
        # Pattern: "Segment" on one line, "XXX -" or "XXX –" on next line
        if line.lower() == "segment" and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Match "TD5 - Name" with digits allowed (like TD5, N1, PO1)
            match = re.match(r"^([A-Z0-9]{2,3})\s+-", next_line)
            if match and match.group(1) in KNOWN_SEGMENTS:
                return True, match.group(1)
            # Match with en-dash
            match = re.match(r"^([A-Z0-9]{2,3})\s+–", next_line)
            if match and match.group(1) in KNOWN_SEGMENTS:
                return True, match.group(1)
    
    return False, None


def detect_toc_pattern(text: str, fonts: List[dict]) -> bool:
    """Detect Table of Contents pages."""
    patterns = [
        r"table\s+of\s+contents",
        r"^contents\s*$",
        r"^\s*contents\s*$",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    # Check fonts for "Contents" or "Table of Contents" in large font
    if fonts:
        max_size = max(f["size"] for f in fonts)
        for f in fonts[:10]:
            if f["size"] >= max_size * 0.9:
                if re.search(r"contents", f["text"], re.IGNORECASE):
                    return True
    
    return False


def detect_glossary_pattern(text: str) -> bool:
    """Detect glossary/abbreviation pages (typically 'other')."""
    patterns = [
        r"^\s*GLOSSARY\s*:?\s*$",
        r"^\s*Abbreviation\s+Meaning",
        r"Document\s+Control",
        r"Change\s+Record",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False


def detect_index_header(text: str) -> bool:
    """Detect index/segment listing page headers."""
    patterns = [
        r"EDI\s+levels\s+and\s+Segments",
        r"Segment\s+Summary",
        r"Segment\s+Directory",
        r"Transaction\s+Set\s+Diagram",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def detect_loop_overview(text: str) -> bool:
    """Detect loop overview/summary pages (not segment definitions)."""
    patterns = [
        r"^Loop\s+Name\s*$",
        r"^Loop\s+[A-Z][a-z]+\s+[A-Z][a-z]+",  # "Loop Baseline Item Data"
        r"Loop\s+ID\s+Name\s+Req",
        r"^\s*Loop:\s*[A-Z]+\s+Name:",
        r"Loop\s+Summary\s*:",  # Loop Summary page
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            # But not if it also has segment definition patterns
            if not re.search(r"Element\s+Summary|Data\s+Element", text, re.IGNORECASE):
                return True
    
    return False


def detect_preface_page(text: str) -> bool:
    """Detect preface/introduction pages (typically 'other')."""
    patterns = [
        r"^\s*Preface\s*$",
        r"Purpose\s+and\s+Scope",
        r"^\s*Introduction\s*$",
        r"^\s*Overview\s*$",
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    return False


def detect_appendix_page(text: str) -> bool:
    """Detect appendix/example pages (typically 'other')."""
    # Check for appendix header
    if re.search(r"^\s*APPENDIX\s*$", text, re.IGNORECASE | re.MULTILINE):
        return True
    
    # Check for transaction example header  
    if re.search(r"Transaction\s+Example", text, re.IGNORECASE):
        return True
    
    # Check for country requirements section
    if re.search(r"Country\s+Specific\s+Requirements", text, re.IGNORECASE):
        return True
    
    return False


def detect_summary_notes(text: str, fonts: List[dict]) -> bool:
    """Detect index continuation pages with Summary/Notes headers."""
    # Check first lines for summary patterns
    # Be careful NOT to match "Element Summary:" which is a segment pattern
    lines = text.strip().split('\n')[:20]
    for line in lines:
        line = line.strip()
        # Match "Summary:" at start but NOT "Element Summary:"
        if re.match(r"^Summary\s*:", line, re.IGNORECASE):
            return True
        if re.match(r"^Notes\s*:", line, re.IGNORECASE):
            return True
        # "Transactions Summary" pattern (index listing)
        if re.search(r"^Transactions?\s+Summary\s*$", line, re.IGNORECASE):
            return True
        # "Transaction Set Notes" pattern (index page)
        if re.search(r"Transaction\s+Set\s+Notes", line, re.IGNORECASE):
            return True
        # "Functional Group=PO" pattern (index page intro)
        if re.search(r"Functional\s+Group\s*=\s*PO", line, re.IGNORECASE):
            return True
    
    return False


def detect_element_table(text: str) -> bool:
    """Detect element definition tables (strong segment indicator)."""
    patterns = [
        r"Element\s+Summary",
        r"Data\s+Element\s+Summary", 
        r"Ref\s+Id\s+Element\s+Name",
        r"\b(M|O|C)\s+(AN|ID|DT|TM|N\d|R)\s+\d+/\d+",  # Element type pattern
    ]
    
    matches = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
    
    return matches >= 2


def compute_scores(features: 'PageFeatures') -> Tuple[float, float, float]:
    """Compute classification scores based on features."""
    
    index_score = 0.0
    segment_score = 0.0
    other_score = 0.0
    
    # INDEX scoring
    if features.has_summary_notes:
        index_score += 5.0
    if features.num_tables > 0 and not features.has_element_table:
        index_score += 2.0
    if features.has_toc_pattern:
        other_score += 5.0  # TOC is actually "other"
    
    # SEGMENT scoring
    if features.has_segment_header and features.segment_id:
        segment_score += 8.0
    if features.has_element_table:
        segment_score += 5.0
    if features.header_font_size > features.avg_font_size * 1.3:
        segment_score += 2.0  # Large header suggests segment start
    
    # Continuation detection (segment page without header)
    # Will be handled by sequence context
    
    # OTHER scoring
    if features.has_toc_pattern:
        other_score += 5.0
    if features.has_loop_overview:
        other_score += 4.0
    if features.text_density < 0.001:
        other_score += 2.0  # Very sparse page
    
    return index_score, segment_score, other_score


def extract_page_features(doc, page_num: int) -> PageFeatures:
    """Extract all features for a single page."""
    page = doc[page_num]
    text = page.get_text()
    
    # Font features
    max_font, min_font, avg_font, std_font, header_font, fonts = extract_font_features(page)
    
    # Content detection
    has_header, segment_id = detect_segment_header(fonts, text)
    has_toc = detect_toc_pattern(text, fonts)
    has_loop = detect_loop_overview(text)
    has_summary = detect_summary_notes(text, fonts)
    has_elements = detect_element_table(text)
    has_glossary = detect_glossary_pattern(text)
    has_index_header = detect_index_header(text)
    has_preface = detect_preface_page(text)
    has_appendix = detect_appendix_page(text)
    
    # Layout features
    tables = page.find_tables()
    num_tables = len(tables.tables) if tables else 0
    
    blocks = page.get_text("dict")["blocks"]
    num_blocks = len([b for b in blocks if b["type"] == 0])
    
    text_density = len(text) / (page.rect.width * page.rect.height) if page.rect.width > 0 else 0
    
    # Create features object
    features = PageFeatures(
        page_num=page_num + 1,
        text=text,
        max_font_size=max_font,
        min_font_size=min_font,
        avg_font_size=avg_font,
        font_size_std=std_font,
        header_font_size=header_font,
        has_segment_header=has_header,
        segment_id=segment_id,
        has_toc_pattern=has_toc,
        has_loop_overview=has_loop,
        has_summary_notes=has_summary,
        has_element_table=has_elements,
        has_glossary=has_glossary,
        has_index_header=has_index_header,
        has_preface=has_preface,
        has_appendix=has_appendix,
        num_tables=num_tables,
        text_density=text_density,
        num_text_blocks=num_blocks,
        index_score=0,
        segment_score=0,
        other_score=0
    )
    
    # Compute scores
    idx_score, seg_score, other_score = compute_scores(features)
    features.index_score = idx_score
    features.segment_score = seg_score
    features.other_score = other_score
    
    return features


def extract_document_features(pdf_path: str) -> List[PageFeatures]:
    """Extract features for all pages in a document."""
    doc = fitz.open(pdf_path)
    features = []
    
    for page_num in range(len(doc)):
        page_features = extract_page_features(doc, page_num)
        features.append(page_features)
    
    doc.close()
    return features
