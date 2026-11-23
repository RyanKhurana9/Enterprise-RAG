"""Smoke test: the Streamlit app renders (and degrades gracefully) without an API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"


def test_renders_with_api_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("Cannot reach API" in e.value for e in at.sidebar.error)
    assert len(at.tabs) == 3 and len(at.chat_input) == 1

    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("Cannot reach API" in e.value for e in at.main.error)
