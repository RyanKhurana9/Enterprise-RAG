from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_rag.ingestion.loaders import MarkdownLoader, PdfLoader, default_loader


def test_markdown_loader(tmp_path: Path) -> None:
    d = tmp_path / "policies"
    d.mkdir()
    p = d / "My Policy_v2.md"
    p.write_text("intro\n# Wire Policy\n\nBody text.\n")
    doc = MarkdownLoader().load(p)
    assert doc.doc_id == "my-policy-v2"
    assert doc.title == "Wire Policy"
    assert doc.doc_type == "policy"
    assert "Body text." in doc.text


def test_pdf_loader_roundtrip(tmp_path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 5, "Liquidity Policy", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, "The LCR floor is 110 percent.", new_x="LMARGIN", new_y="NEXT")
    p = tmp_path / "10k" / "liq.pdf"
    p.parent.mkdir()
    pdf.output(str(p))
    doc = PdfLoader().load(p)
    assert doc.title == "Liquidity Policy"
    assert "110 percent" in doc.text
    assert doc.doc_type == "10k"


def test_composite_discovers_and_dispatches(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    (tmp_path / "b.bin").write_bytes(b"\x00")
    loader = default_loader()
    found = list(loader.discover(tmp_path))
    assert [p.name for p in found] == ["a.md"]
    assert loader.load(found[0]).title == "A"
    with pytest.raises(ValueError):
        loader.load(tmp_path / "b.bin")
