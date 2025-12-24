"""
Generate predictions in ground truth format.
Output format matches data/sample_docs/page_splits_new/850_gt/*.json
"""

import json
from pathlib import Path
from src.labeling_rules import label_document

PDF_DIR = "data/sample_docs/x12_specs/850"
OUTPUT_DIR = "predictions"


def generate_prediction(pdf_path: str) -> dict:
    """Generate prediction in ground truth format."""
    results = label_document(pdf_path)
    
    pdf_name = Path(pdf_path).stem
    total_pages = len(results)
    
    # Collect pages by type
    segment_table_pages = []  # index pages
    other_pages = []
    segment_pages = {}  # segment_id -> [pages]
    
    for i, result in enumerate(results):
        page_num = i + 1
        
        if result.label == "index":
            segment_table_pages.append(page_num)
        elif result.label == "other":
            other_pages.append(page_num)
        elif result.label == "segment":
            seg_id = result.segment_id or "UNKNOWN"
            if seg_id not in segment_pages:
                segment_pages[seg_id] = []
            segment_pages[seg_id].append(page_num)
    
    # Build output in GT format
    output = {
        "filename": f"{pdf_name}.pdf",
        "total_pages": total_pages,
        "segment_table_pages": segment_table_pages,
        "segment_pages": [
            {"segment": seg_id, "pages": pages}
            for seg_id, pages in segment_pages.items()
        ],
        "other_pages": other_pages
    }
    
    return output


def main():
    pdf_dir = Path(PDF_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        print(f"Processing {pdf_file.name}...")
        try:
            prediction = generate_prediction(str(pdf_file))
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(prediction, f, indent=4)
            
            print(f"  -> {output_file}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
