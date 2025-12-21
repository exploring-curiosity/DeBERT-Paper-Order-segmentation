"""
Deep analysis of PDF documents against ground truth.
Extracts font sizes, layout patterns, and structural features.
"""

import json
import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import re


@dataclass
class FontInfo:
    """Font information for text blocks."""
    size: float
    flags: int  # bold, italic, etc.
    name: str
    text: str


@dataclass 
class PageAnalysis:
    """Analysis results for a single page."""
    page_num: int
    fonts: List[FontInfo]
    max_font_size: float
    min_font_size: float
    avg_font_size: float
    header_text: str  # Text in largest font
    has_segment_header: bool
    segment_id: Optional[str]
    text_density: float
    num_tables: int
    layout_type: str  # 'single_column', 'multi_column', 'table'


def extract_font_info(page) -> List[FontInfo]:
    """Extract font information from all text blocks on a page."""
    fonts = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        fonts.append(FontInfo(
                            size=span["size"],
                            flags=span["flags"],
                            name=span["font"],
                            text=span["text"].strip()
                        ))
    return fonts


def analyze_page(doc, page_num: int) -> PageAnalysis:
    """Analyze a single page for structural features."""
    page = doc[page_num]
    fonts = extract_font_info(page)
    
    if not fonts:
        return PageAnalysis(
            page_num=page_num + 1,
            fonts=[],
            max_font_size=0,
            min_font_size=0,
            avg_font_size=0,
            header_text="",
            has_segment_header=False,
            segment_id=None,
            text_density=0,
            num_tables=0,
            layout_type="empty"
        )
    
    # Font size statistics
    sizes = [f.size for f in fonts]
    max_size = max(sizes)
    min_size = min(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    # Find header text (largest font in top portion)
    header_fonts = [f for f in fonts[:20] if f.size >= max_size * 0.9]
    header_text = " ".join(f.text for f in header_fonts[:5])
    
    # Check for segment header pattern
    segment_id = None
    has_segment_header = False
    
    # Known X12 segments
    known_segments = [
        "ISA", "GS", "ST", "BEG", "CUR", "REF", "PER", "TAX", "FOB", "ITD",
        "DTM", "DIS", "INC", "LIN", "SI", "PID", "MEA", "PWK", "PKG", "TD1",
        "TD5", "TD3", "TD4", "MAN", "PCT", "CTB", "TXI", "SAC", "CTP", "PAM",
        "N9", "N1", "N2", "N3", "N4", "PO1", "PO3", "PO4", "CTP", "CTT", "AMT",
        "SE", "GE", "IEA", "CSH", "TC2", "SLN", "SDQ", "SCH", "MSG", "SPO", "ADV"
    ]
    
    for f in fonts[:30]:
        # Check if this is a segment ID in large font
        text = f.text.strip()
        for seg in known_segments:
            if text == seg or text.startswith(seg + " ") or text.startswith(seg + "\n"):
                if f.size >= avg_size * 1.1:  # Larger than average
                    segment_id = seg
                    has_segment_header = True
                    break
        
        # Also check pattern "Segment: XXX"
        match = re.search(r"Segment:\s*([A-Z]{2,3})", text)
        if match:
            segment_id = match.group(1)
            has_segment_header = True
            break
    
    # Text density
    text = page.get_text()
    text_density = len(text) / (page.rect.width * page.rect.height) if page.rect.width > 0 else 0
    
    # Detect tables
    tables = page.find_tables()
    num_tables = len(tables.tables) if tables else 0
    
    # Layout type detection
    blocks = page.get_text("dict")["blocks"]
    if num_tables > 0:
        layout_type = "table"
    elif len(blocks) > 10:
        layout_type = "multi_column"
    else:
        layout_type = "single_column"
    
    return PageAnalysis(
        page_num=page_num + 1,
        fonts=fonts,
        max_font_size=max_size,
        min_font_size=min_size,
        avg_font_size=avg_size,
        header_text=header_text,
        has_segment_header=has_segment_header,
        segment_id=segment_id,
        text_density=text_density,
        num_tables=num_tables,
        layout_type=layout_type
    )


def load_ground_truth(gt_path: Path) -> Dict:
    """Load ground truth JSON."""
    with open(gt_path) as f:
        return json.load(f)


def get_page_label_from_gt(gt: Dict, page_num: int) -> Tuple[str, Optional[str]]:
    """Get the label and segment ID for a page from ground truth."""
    # Check if index page
    if page_num in gt.get("segment_table_pages", []):
        return "index", None
    
    # Check if segment page
    for seg_info in gt.get("segment_pages", []):
        if page_num in seg_info.get("pages", []):
            return "segment", seg_info["segment"]
    
    return "other", None


def analyze_document(pdf_path: Path, gt_path: Path) -> Dict:
    """Analyze a document against its ground truth."""
    doc = fitz.open(str(pdf_path))
    gt = load_ground_truth(gt_path)
    
    results = {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "pages": [],
        "patterns": {
            "index": {"font_sizes": [], "has_tables": [], "text_densities": []},
            "segment": {"font_sizes": [], "has_header": [], "segment_ids": []},
            "segment_continuation": {"font_sizes": [], "has_header": []},
            "other": {"font_sizes": [], "text_densities": []}
        }
    }
    
    prev_segment = None
    
    for page_num in range(len(doc)):
        analysis = analyze_page(doc, page_num)
        gt_label, gt_segment = get_page_label_from_gt(gt, page_num + 1)
        
        # Track if this is a continuation page
        is_continuation = False
        if gt_label == "segment" and gt_segment:
            seg_pages = None
            for seg_info in gt.get("segment_pages", []):
                if seg_info["segment"] == gt_segment:
                    seg_pages = seg_info["pages"]
                    break
            if seg_pages and page_num + 1 != seg_pages[0]:
                is_continuation = True
        
        page_result = {
            "page_num": page_num + 1,
            "gt_label": gt_label,
            "gt_segment": gt_segment,
            "is_continuation": is_continuation,
            "max_font_size": analysis.max_font_size,
            "avg_font_size": analysis.avg_font_size,
            "has_segment_header": analysis.has_segment_header,
            "detected_segment": analysis.segment_id,
            "header_text": analysis.header_text[:100] if analysis.header_text else "",
            "num_tables": analysis.num_tables,
            "text_density": analysis.text_density,
            "layout_type": analysis.layout_type
        }
        results["pages"].append(page_result)
        
        # Collect patterns by label
        if gt_label == "index":
            results["patterns"]["index"]["font_sizes"].append(analysis.max_font_size)
            results["patterns"]["index"]["has_tables"].append(analysis.num_tables > 0)
            results["patterns"]["index"]["text_densities"].append(analysis.text_density)
        elif gt_label == "segment":
            if is_continuation:
                results["patterns"]["segment_continuation"]["font_sizes"].append(analysis.max_font_size)
                results["patterns"]["segment_continuation"]["has_header"].append(analysis.has_segment_header)
            else:
                results["patterns"]["segment"]["font_sizes"].append(analysis.max_font_size)
                results["patterns"]["segment"]["has_header"].append(analysis.has_segment_header)
                results["patterns"]["segment"]["segment_ids"].append(gt_segment)
        else:
            results["patterns"]["other"]["font_sizes"].append(analysis.max_font_size)
            results["patterns"]["other"]["text_densities"].append(analysis.text_density)
    
    doc.close()
    return results


def analyze_corpus(pdf_dir: Path, gt_dir: Path) -> Dict:
    """Analyze all documents in the corpus."""
    all_results = []
    
    for gt_file in sorted(gt_dir.glob("*.json")):
        stem = gt_file.stem
        pdf_file = pdf_dir / f"{stem}.pdf"
        
        if pdf_file.exists():
            print(f"Analyzing {stem}...")
            result = analyze_document(pdf_file, gt_file)
            all_results.append(result)
    
    return all_results


def summarize_patterns(results: List[Dict]) -> Dict:
    """Summarize patterns across all documents."""
    summary = {
        "index": {
            "avg_max_font": [],
            "table_ratio": [],
            "avg_density": []
        },
        "segment_first_page": {
            "avg_max_font": [],
            "header_detection_rate": [],
            "segment_ids": Counter()
        },
        "segment_continuation": {
            "avg_max_font": [],
            "header_detection_rate": []
        },
        "other": {
            "avg_max_font": [],
            "avg_density": []
        },
        "errors": []
    }
    
    for doc in results:
        patterns = doc["patterns"]
        
        if patterns["index"]["font_sizes"]:
            summary["index"]["avg_max_font"].extend(patterns["index"]["font_sizes"])
            summary["index"]["table_ratio"].extend(patterns["index"]["has_tables"])
            summary["index"]["avg_density"].extend(patterns["index"]["text_densities"])
        
        if patterns["segment"]["font_sizes"]:
            summary["segment_first_page"]["avg_max_font"].extend(patterns["segment"]["font_sizes"])
            summary["segment_first_page"]["header_detection_rate"].extend(patterns["segment"]["has_header"])
            for seg in patterns["segment"]["segment_ids"]:
                summary["segment_first_page"]["segment_ids"][seg] += 1
        
        if patterns["segment_continuation"]["font_sizes"]:
            summary["segment_continuation"]["avg_max_font"].extend(patterns["segment_continuation"]["font_sizes"])
            summary["segment_continuation"]["header_detection_rate"].extend(patterns["segment_continuation"]["has_header"])
        
        if patterns["other"]["font_sizes"]:
            summary["other"]["avg_max_font"].extend(patterns["other"]["font_sizes"])
            summary["other"]["avg_density"].extend(patterns["other"]["text_densities"])
        
        # Check for detection errors
        for page in doc["pages"]:
            if page["gt_label"] == "segment" and not page["is_continuation"]:
                if not page["has_segment_header"]:
                    summary["errors"].append({
                        "file": doc["filename"],
                        "page": page["page_num"],
                        "issue": "segment_header_not_detected",
                        "expected_segment": page["gt_segment"],
                        "header_text": page["header_text"]
                    })
            elif page["gt_label"] == "segment" and page["is_continuation"]:
                if page["has_segment_header"]:
                    summary["errors"].append({
                        "file": doc["filename"],
                        "page": page["page_num"],
                        "issue": "continuation_has_header",
                        "detected_segment": page["detected_segment"]
                    })
    
    return summary


if __name__ == "__main__":
    pdf_dir = Path("data/sample_docs/x12_specs/850")
    gt_dir = Path("data/sample_docs/page_splits/850_gt")
    
    print("=" * 70)
    print("PDF CORPUS DEEP ANALYSIS")
    print("=" * 70)
    
    results = analyze_corpus(pdf_dir, gt_dir)
    summary = summarize_patterns(results)
    
    print("\n" + "=" * 70)
    print("PATTERN SUMMARY")
    print("=" * 70)
    
    # Index pages
    print("\n### INDEX PAGES ###")
    if summary["index"]["avg_max_font"]:
        print(f"  Avg max font size: {sum(summary['index']['avg_max_font'])/len(summary['index']['avg_max_font']):.1f}")
        print(f"  Has tables ratio: {sum(summary['index']['table_ratio'])/len(summary['index']['table_ratio'])*100:.1f}%")
        print(f"  Avg text density: {sum(summary['index']['avg_density'])/len(summary['index']['avg_density']):.4f}")
    
    # Segment first pages
    print("\n### SEGMENT FIRST PAGES ###")
    if summary["segment_first_page"]["avg_max_font"]:
        print(f"  Avg max font size: {sum(summary['segment_first_page']['avg_max_font'])/len(summary['segment_first_page']['avg_max_font']):.1f}")
        header_rate = sum(summary['segment_first_page']['header_detection_rate'])/len(summary['segment_first_page']['header_detection_rate'])*100
        print(f"  Header detection rate: {header_rate:.1f}%")
        print(f"  Most common segments: {summary['segment_first_page']['segment_ids'].most_common(10)}")
    
    # Segment continuations
    print("\n### SEGMENT CONTINUATION PAGES ###")
    if summary["segment_continuation"]["avg_max_font"]:
        print(f"  Avg max font size: {sum(summary['segment_continuation']['avg_max_font'])/len(summary['segment_continuation']['avg_max_font']):.1f}")
        cont_header_rate = sum(summary['segment_continuation']['header_detection_rate'])/len(summary['segment_continuation']['header_detection_rate'])*100
        print(f"  Has segment header: {cont_header_rate:.1f}%")
    
    # Other pages
    print("\n### OTHER PAGES ###")
    if summary["other"]["avg_max_font"]:
        print(f"  Avg max font size: {sum(summary['other']['avg_max_font'])/len(summary['other']['avg_max_font']):.1f}")
        print(f"  Avg text density: {sum(summary['other']['avg_density'])/len(summary['other']['avg_density']):.4f}")
    
    # Detection errors
    print("\n### DETECTION ISSUES ###")
    print(f"  Total issues: {len(summary['errors'])}")
    for err in summary["errors"][:20]:
        print(f"  - {err['file']} p{err['page']}: {err['issue']}")
        if "expected_segment" in err:
            print(f"      Expected: {err['expected_segment']}, Header: {err['header_text'][:50]}")
    
    # Save detailed results
    output_file = Path("analysis/corpus_analysis.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_file}")
