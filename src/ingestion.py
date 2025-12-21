"""
Robust PDF Ingestion Module.
Handles text extraction from native and scanned PDFs.
"""

import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int  # 0-indexed
    text: str
    width: float
    height: float
    is_scanned: bool = False
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())


def compute_text_density(page: fitz.Page) -> float:
    """
    Compute text density as ratio of text area to page area.
    Low density suggests scanned/image-based page.
    """
    text_area = 0.0
    total_area = page.rect.width * page.rect.height
    
    if total_area == 0:
        return 0.0
    
    text_blocks = page.get_text("blocks")
    for block in text_blocks:
        if len(block) >= 5 and block[4].strip():
            r = fitz.Rect(block[:4])
            text_area += r.width * r.height
        
    return text_area / total_area


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """Run OCR on a PDF page using Tesseract."""
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed, skipping OCR")
        return ""
    
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    
    text = pytesseract.image_to_string(image)
    return text


def ingest_pdf(
    pdf_path: str, 
    use_ocr: bool = True,
    text_density_threshold: float = 0.02
) -> List[PageContent]:
    """
    Ingest a PDF file and extract text from all pages.
    Automatically falls back to OCR if pages appear scanned.
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to use OCR for scanned pages
        text_density_threshold: Below this density, page is considered scanned
    
    Returns:
        List of PageContent objects for each page
    """
    doc = fitz.open(pdf_path)
    contents = []
    
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text()
        is_scanned = False
        
        # Check if scanned based on text density
        density = compute_text_density(page)
        needs_ocr = use_ocr and (not text.strip() or density < text_density_threshold)
        
        if needs_ocr:
            logger.debug(f"Page {i+1} in {pdf_path} appears scanned (density={density:.3f}). Running OCR...")
            try:
                ocr_text = ocr_page(page)
                if ocr_text.strip():
                    text = ocr_text
                    is_scanned = True
            except Exception as e:
                logger.warning(f"OCR failed for page {i+1} in {pdf_path}: {e}")
        
        contents.append(PageContent(
            page_number=i,
            text=text,
            width=page.rect.width,
            height=page.rect.height,
            is_scanned=is_scanned
        ))
        
    doc.close()
    return contents


def ingest_pdf_batch(
    pdf_paths: List[str],
    use_ocr: bool = True,
    show_progress: bool = True
) -> dict:
    """
    Ingest multiple PDFs and return a dictionary mapping filename to pages.
    """
    from tqdm import tqdm
    from pathlib import Path
    
    results = {}
    iterator = tqdm(pdf_paths, desc="Ingesting PDFs") if show_progress else pdf_paths
    
    for pdf_path in iterator:
        try:
            pages = ingest_pdf(pdf_path, use_ocr=use_ocr)
            filename = Path(pdf_path).stem
            results[filename] = pages
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path}: {e}")
            
    return results
