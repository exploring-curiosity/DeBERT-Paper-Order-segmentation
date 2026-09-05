"""
Test the enhanced labeling system against ground truth.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labeling_rules import (
    label_document,
    LABEL_INDEX,
    LABEL_SEGMENT,
    LABEL_OTHER,
    LabelResult,
    classify_page,
    has_segment_listing_table,
    has_element_definition_structure,
    has_segment_label_header,
    is_raw_edi_sample,
    validate_document_structure,
    generate_labels,
)
from src.features import PageFeatures, extract_document_features, KNOWN_SEGMENTS


# =============================================================================
# Constants and module-level tests
# =============================================================================


class TestLabelConstants:
    """Test that labeling constants are properly defined."""

    def test_label_index_exists(self):
        assert LABEL_INDEX == "index"

    def test_label_segment_exists(self):
        assert LABEL_SEGMENT == "segment"

    def test_label_other_exists(self):
        assert LABEL_OTHER == "other"

    def test_labels_dict(self):
        from src.labeling_rules import LABELS, LABEL_NAMES
        assert LABELS == {LABEL_INDEX: 0, LABEL_SEGMENT: 1, LABEL_OTHER: 2}
        assert LABEL_NAMES == {0: LABEL_INDEX, 1: LABEL_SEGMENT, 2: LABEL_OTHER}

    def test_known_segments_nonempty(self):
        assert len(KNOWN_SEGMENTS) > 0


# =============================================================================
# LabelResult dataclass tests
# =============================================================================


class TestLabelResult:
    """Test the LabelResult dataclass."""

    def test_basic_result(self):
        result = LabelResult(LABEL_OTHER, 0.5, reason="default")
        assert result.label == LABEL_OTHER
        assert result.confidence == 0.5
        assert result.segment_id is None
        assert result.reason == "default"
        assert result.is_continuation is False

    def test_result_with_segment(self):
        result = LabelResult(
            LABEL_SEGMENT, 0.98, segment_id="ST", reason="segment_label_header"
        )
        assert result.label == LABEL_SEGMENT
        assert result.segment_id == "ST"
        assert result.confidence == 0.98

    def test_result_continuation(self):
        result = LabelResult(
            LABEL_SEGMENT, 0.70, segment_id="BEG", reason="structure_segment",
            is_continuation=True
        )
        assert result.is_continuation is True


# =============================================================================
# Pattern detection function tests
# =============================================================================


class TestPatternDetection:
    """Test the structural pattern detection functions."""

    def test_has_segment_listing_table_finds_table(self):
        text = """0100  ST  Transaction Set Header
0200  BEG  Beginning Segment
0300  REF  Reference Identification
0400  PER  Contact Information"""
        assert has_segment_listing_table(text) is True

    def test_has_segment_listing_table_no_table(self):
        text = "This is just regular text without any tabular data."
        assert has_segment_listing_table(text) is False

    def test_has_segment_listing_table_few_entries(self):
        text = "0100  ST  Transaction Set Header"
        assert has_segment_listing_table(text) is False

    def test_has_element_definition_structure(self):
        text = "Pos: 010\nM AN 1/30\nUsage: Mandatory"
        assert has_element_definition_structure(text) is True

    def test_has_element_definition_structure_no_match(self):
        text = "This is just regular text."
        assert has_element_definition_structure(text) is False

    def test_has_segment_label_header(self):
        text = "Segment ST – Transaction Set Header"
        found, seg_id = has_segment_label_header(text)
        assert found is True
        assert seg_id == "ST"

    def test_has_segment_label_header_no_match(self):
        text = "This is just regular text."
        found, seg_id = has_segment_label_header(text)
        assert found is False
        assert seg_id is None

    def test_is_raw_edi_sample(self):
        text = "ST*850*0001~\nBE*00*NE*12345**20231224~\nPO*01*12345~\nIT*01*12345~"
        assert is_raw_edi_sample(text) is True

    def test_is_raw_edi_sample_no_edi(self):
        text = "This is just regular text without EDI data."
        assert is_raw_edi_sample(text) is False


# =============================================================================
# classify_page tests
# =============================================================================


def _make_features(text: str, **kwargs):
    """Helper to create a PageFeatures object with all required fields."""
    return PageFeatures(
        page_num=kwargs.get("page_num", 1),
        text=text,
        max_font_size=kwargs.get("max_font_size", 12.0),
        min_font_size=kwargs.get("min_font_size", 8.0),
        avg_font_size=kwargs.get("avg_font_size", 10.0),
        font_size_std=kwargs.get("font_size_std", 1.0),
        header_font_size=kwargs.get("header_font_size", 12.0),
        has_segment_header=kwargs.get("has_segment_header", False),
        segment_id=kwargs.get("segment_id", None),
        has_toc_pattern=kwargs.get("has_toc_pattern", False),
        has_loop_overview=kwargs.get("has_loop_overview", False),
        has_summary_notes=kwargs.get("has_summary_notes", False),
        has_element_table=kwargs.get("has_element_table", False),
        has_glossary=kwargs.get("has_glossary", False),
        has_index_header=kwargs.get("has_index_header", False),
        has_preface=kwargs.get("has_preface", False),
        has_appendix=kwargs.get("has_appendix", False),
        num_tables=kwargs.get("num_tables", 0),
        text_density=kwargs.get("text_density", 0.5),
        num_text_blocks=kwargs.get("num_text_blocks", 10),
        index_score=kwargs.get("index_score", 0.0),
        segment_score=kwargs.get("segment_score", 0.0),
        other_score=kwargs.get("other_score", 0.0),
    )


class TestClassifyPage:
    """Test the page classification logic."""

    def test_glossary_detected_as_other(self):
        features = _make_features("Glossary of terms", has_glossary=True)
        result = classify_page(features)
        assert result.label == LABEL_OTHER
        assert result.reason == "glossary"

    def test_preface_detected_as_other(self):
        features = _make_features("Preface", has_preface=True)
        result = classify_page(features)
        assert result.label == LABEL_OTHER
        assert result.reason == "preface"

    def test_appendix_detected_as_other(self):
        features = _make_features("Appendix A", has_appendix=True)
        result = classify_page(features)
        assert result.label == LABEL_OTHER
        assert result.reason == "appendix"

    def test_segment_label_header(self):
        features = _make_features("Segment ST – Transaction Set Header")
        result = classify_page(features)
        assert result.label == LABEL_SEGMENT
        assert result.segment_id == "ST"

    def test_default_fallback(self):
        features = _make_features("Some regular text content here.")
        result = classify_page(features)
        assert result.label == LABEL_OTHER
        assert result.reason == "default"


# =============================================================================
# validate_document_structure tests
# =============================================================================


class TestValidateDocumentStructure:
    """Test document structure validation."""

    def test_empty_results(self):
        from src.labeling_rules import validate_document_structure
        result = validate_document_structure([], [])
        assert result == []

    def test_basic_structure_preservation(self):
        from src.labeling_rules import validate_document_structure

        results = [
            LabelResult(LABEL_OTHER, 0.95, reason="preface"),
            LabelResult(LABEL_INDEX, 0.92, reason="index"),
            LabelResult(LABEL_SEGMENT, 0.98, segment_id="ST", reason="segment_label_header"),
            LabelResult(LABEL_SEGMENT, 0.85, segment_id="BEG", reason="segment_patterns"),
        ]
        features_list = [
            _make_features("Preface", page_num=1),
            _make_features("Segment listing", page_num=2),
            _make_features("Segment ST definition", page_num=3),
            _make_features("Segment BEG definition", page_num=4),
        ]
        validated = validate_document_structure(results, features_list)
        assert len(validated) == 4


# =============================================================================
# Integration tests (require data, skip if unavailable)
# =============================================================================


DATA_DIR = Path("data/sample_docs")
PDF_DIR = DATA_DIR / "x12_specs/850"
GT_DIR = DATA_DIR / "page_splits/850_gt"

HAS_DATA = PDF_DIR.exists() and GT_DIR.exists()


def load_ground_truth(gt_dir: Path) -> dict:
    """Load all ground truth files."""
    gt_data = {}
    for gt_file in gt_dir.glob("*.json"):
        with open(gt_file) as f:
            data = json.load(f)
            gt_data[gt_file.stem] = data
    return gt_data


def get_gt_label(gt: dict, page_num: int) -> tuple:
    """Get ground truth label and segment for a page."""
    if page_num in gt.get("segment_table_pages", []):
        return "index", None

    for seg in gt.get("segment_pages", []):
        if page_num in seg["pages"]:
            return "segment", seg["segment"]

    return "other", None


def evaluate_labeling(pdf_dir: Path, gt_dir: Path):
    """Evaluate labeling accuracy against ground truth."""

    gt_data = load_ground_truth(gt_dir)

    total = 0
    correct = 0

    errors_by_type = defaultdict(list)
    doc_results = {}

    for stem, gt in sorted(gt_data.items()):
        pdf_path = pdf_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"PDF not found: {pdf_path}")
            continue

        try:
            results = label_document(str(pdf_path))
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            continue

        doc_correct = 0
        doc_total = gt.get("total_pages", len(results))

        for i, result in enumerate(results):
            page_num = i + 1
            total += 1

            gt_label, gt_segment = get_gt_label(gt, page_num)
            pred_label = result.label

            if pred_label == gt_label:
                correct += 1
                doc_correct += 1
            else:
                error_key = f"{gt_label}_as_{pred_label}"
                errors_by_type[error_key].append({
                    "file": stem,
                    "page": page_num,
                    "predicted": pred_label,
                    "actual": gt_label,
                    "segment": gt_segment,
                    "confidence": result.confidence,
                    "reason": result.reason
                })

        doc_acc = doc_correct / doc_total if doc_total > 0 else 0
        doc_results[stem] = doc_acc
        print(f"{stem}: {doc_correct}/{doc_total} ({doc_acc*100:.1f}%)")

    print("\n" + "=" * 70)
    print("LABELING ACCURACY (vs Ground Truth)")
    print("=" * 70)
    if total > 0:
        print(f"\nOverall: {correct}/{total} ({correct/total*100:.2f}%)")
    else:
        print("\nOverall: No pages evaluated (no data)")

    for error_type, errors in sorted(errors_by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n{error_type}: {len(errors)} cases")
        for e in errors[:5]:
            print(f"  {e['file']} p{e['page']}: reason={e['reason']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print("\n" + "-" * 50)
    print("DOCUMENTS NEEDING ATTENTION")
    print("-" * 50)

    for stem, acc in sorted(doc_results.items(), key=lambda x: x[1]):
        if acc < 1.0:
            print(f"  {stem}: {acc*100:.1f}%")

    return correct, total, errors_by_type


@pytest.mark.skipif(not HAS_DATA, reason="Test data not available (data/sample_docs missing)")
class TestLabelingIntegration:
    """Integration tests that require PDF and ground truth data."""

    def test_label_document_returns_results(self):
        """label_document should return a list of LabelResult for any valid PDF."""
        # Just verify the function is callable and returns a list
        results = label_document(str(PDF_DIR / "test.pdf"))
        assert isinstance(results, list)

    def test_evaluate_labeling_runs(self):
        """evaluate_labeling should complete without error when data exists."""
        correct, total, errors = evaluate_labeling(PDF_DIR, GT_DIR)
        assert isinstance(correct, int)
        assert isinstance(total, int)
        assert isinstance(errors, dict)

    def test_label_document_returns_label_results(self):
        """Each result from label_document should be a LabelResult."""
        results = label_document(str(PDF_DIR / "test.pdf"))
        for r in results:
            assert isinstance(r, LabelResult)
            assert r.label in (LABEL_INDEX, LABEL_SEGMENT, LABEL_OTHER)
