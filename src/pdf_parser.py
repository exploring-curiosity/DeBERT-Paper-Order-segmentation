"""
PDF Parser for 850 Purchase Order documents.
Extracts text and metadata from PDF pages for classification.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class PageData:
    """Represents extracted data from a single PDF page."""
    page_number: int
    text: str
    has_segment_header: bool
    segment_name: Optional[str]
    has_table_structure: bool
    word_count: int
    
    
def extract_segment_name(text: str) -> Optional[str]:
    """Extract segment name from page text if present."""
    # Pattern 1: "Segment: XX Segment Name" (AmerisourceBergen format)
    pattern1 = r"Segment:\s*([A-Z0-9]+)\s+([A-Za-z0-9\s\-/]+?)(?:\s{2,}|Position:|$)"
    match = re.search(pattern1, text)
    if match:
        segment_id = match.group(1).strip()
        segment_desc = match.group(2).strip()
        return f"{segment_id} - {segment_desc}"
    
    # Pattern 2: "XX\nSegment Name\nPos:" (ERICO format)
    pattern2 = r"^\s*([A-Z][A-Z0-9]{0,3})\s*\n([A-Za-z][A-Za-z0-9\s\-/]+?)\s*\n.*?Pos:"
    match = re.search(pattern2, text, re.MULTILINE | re.DOTALL)
    if match:
        segment_id = match.group(1).strip()
        segment_desc = match.group(2).strip()
        # Clean up multi-line descriptions
        segment_desc = ' '.join(segment_desc.split())
        return f"{segment_id} - {segment_desc}"
    
    # Pattern 3: Look for segment ID at start of line followed by description
    pattern3 = r"^\s{0,5}([A-Z][A-Z0-9]{1,3})\s{2,}([A-Za-z][A-Za-z0-9\s\-/]+?)\s*$"
    lines = text.split('\n')
    for i, line in enumerate(lines[:20]):  # Check first 20 lines
        match = re.match(pattern3, line)
        if match:
            segment_id = match.group(1).strip()
            segment_desc = match.group(2).strip()
            # Verify it looks like a segment header (next lines have Pos: or similar)
            remaining = '\n'.join(lines[i+1:i+5])
            if re.search(r'Pos:|Position:|Max:|Loop:', remaining, re.IGNORECASE):
                return f"{segment_id} - {segment_desc}"
    
    return None


def has_index_table_structure(text: str) -> bool:
    """Check if page has index/summary table structure."""
    # Index pages have columns like: Page No., Pos. No., Seg. ID, Name, etc.
    index_indicators = [
        r"Page\s+Pos\.\s+Seg\.",
        r"No\.\s+No\.\s+ID\s+Name",
        r"Base\s+User",
        r"Max\.Use\s+Repeat",
        r"LOOP ID",
        r"Summary:",
        r"Heading:",
        r"Detail:",
        # Additional patterns for different formats
        r"Table of Contents",
        r"Pos\s+Id\s+Segment\s+Name",
        r"Req\s+Max\s+Use\s+Repeat",
        r"Functional\s+Group\s*=",
    ]
    count = sum(1 for pattern in index_indicators if re.search(pattern, text, re.IGNORECASE))
    return count >= 2


def has_segment_structure(text: str) -> bool:
    """Check if page has segment definition structure."""
    segment_indicators = [
        r"Segment:\s+[A-Z0-9]+",
        r"Position:\s+\d+",
        r"Loop:",
        r"Level:\s+(Heading|Detail|Summary)",
        r"Usage:\s+(Mandatory|Optional|Conditional)",
        r"Max Use:",
        r"Data Element Summary",
        r"Ref\.\s+Data",
        r"Des\.\s+Element",
        # Additional patterns for different formats (ERICO style)
        r"Pos:\s*\d+",
        r"Max:\s*\d+",
        r"Element\s+Summary",
        r"User\s+Option\s+\(Usage\)",
        r"(Heading|Detail|Summary)\s*-\s*(Mandatory|Optional)",
        r"Elements:\s*\d+",
    ]
    count = sum(1 for pattern in segment_indicators if re.search(pattern, text, re.IGNORECASE))
    return count >= 3


def parse_pdf(pdf_path: str) -> List[PageData]:
    """Parse PDF and extract page data for classification."""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # Extract features
        segment_name = extract_segment_name(text)
        has_segment = has_segment_structure(text)
        has_table = has_index_table_structure(text)
        word_count = len(text.split())
        
        page_data = PageData(
            page_number=page_num,
            text=text,
            has_segment_header=has_segment,
            segment_name=segment_name,
            has_table_structure=has_table,
            word_count=word_count
        )
        pages.append(page_data)
    
    doc.close()
    return pages


def get_page_images(pdf_path: str, dpi: int = 150) -> List:
    """Extract page images from PDF for visual classification."""
    from PIL import Image
    import io
    
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    
    doc.close()
    return images


if __name__ == "__main__":
    # Test parsing
    pdf_path = "data/850---purchase-order-regular-and-drop-ship_updated.pdf"
    pages = parse_pdf(pdf_path)
    
    for page in pages:
        print(f"Page {page.page_number}: "
              f"segment={page.has_segment_header}, "
              f"table={page.has_table_structure}, "
              f"name={page.segment_name}")
