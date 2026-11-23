"""Streamlit front-end for the Enterprise RAG API.

    streamlit run ui/streamlit_app.py          # make ui

The UI is a thin client: everything goes through the FastAPI service (``RAG_API_URL``,
default http://localhost:8000), so it shows exactly what API consumers get.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
QUERY_TIMEOUT_S = float(os.environ.get("RAG_UI_TIMEOUT_S", "180"))

EXAMPLES: dict[str, list[str]] = {
    "Policies": [
        "What ownership stake makes someone a beneficial owner that Corporate Banking must verify?",
        "What flight duration lets Corporate Banking staff fly in a premium cabin?",
        "What pattern of cash deposits triggers a structuring investigation in Wealth Management?",
    ],
    "10-K filings": [
        "Where is Barnett Financial headquartered?",
        "What did Riley Energy spend on capex in fiscal 2024?",
    ],
    "Employee FAQ": [
        "How much money per year can a United Kingdom employee use for professional courses?",
        "How long does a password reset email remain usable for Singapore users?",
    ],
    "Trading database": [
        "How many trades did EMEA clients make in 2024?",
        "What was the average trade price of MSFT in 2023?",
        "Which 5 tickers have the highest total notional?",
        "How many accounts have fewer than 5 trades?",
    ],
    "Hybrid": [
        "Do any single trades exceed the Global Markets VaR limit?",
    ],
}
ROUTE_COLORS = {"docs": "blue", "sql": "green", "hybrid": "violet"}
CACHE_COLORS = {"exact": "orange", "semantic": "orange", "miss": "gray", "bypass": "gray"}

st.set_page_config(page_title="Enterprise RAG", page_icon="🔎", layout="wide")


# ------------------------------------------------------------------------- API client ----
@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=httpx.Timeout(QUERY_TIMEOUT_S, connect=3.0))


def api(method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        resp = _client().request(method, path, **kwargs)
    except httpx.HTTPError as e:
        return None, f"Cannot reach API at {API_URL}: {type(e).__name__}"
    try:
        body = resp.json()
    except ValueError:
        body = {"detail": resp.text[:300]}
    if resp.status_code >= 400 and resp.status_code != 503:
        return None, str(body.get("detail", body))
    return body, None


@st.cache_data(ttl=10, show_spinner=False)
def get_health() -> tuple[dict[str, Any] | None, str | None]:
    return api("GET", "/health")


@st.cache_data(ttl=10, show_spinner=False)
def get_stats() -> tuple[dict[str, Any] | None, str | None]:
    return api("GET", "/stats")


# ---------------------------------------------------------------------------- state ----
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)


# -------------------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.title("🔎 Enterprise RAG")
    st.caption(f"API: `{API_URL}`")

    health, herr = get_health()
    if herr:
        st.error(herr)
    elif health:
        overall = health["status"]
        icon = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(overall, "⚪")
        st.markdown(f"**System status:** {icon} {overall}")
        for name, check in health["checks"].items():
            ok = check["status"] == "ok"
            st.markdown(
                f"{'✅' if ok else '⚠️'} `{name}` · {check['ms']:.0f} ms"
                + ("" if ok else f"  \n<small>{check['status']}</small>"),
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Query options")
    route_label = st.radio(
        "Route",
        ["Auto", "Docs", "SQL", "Hybrid"],
        horizontal=True,
        help="Auto lets the router decide between documents, the trading database or both.",
    )
    use_cache = st.toggle("Use response cache", value=True)
    top_k = st.slider("Sources given to the LLM (top-k)", 1, 10, 5)
    show_debug = st.toggle("Show timings & debug", value=True)

    stats, serr = get_stats()
    if stats:
        st.divider()
        st.subheader("Corpus")
        c1, c2 = st.columns(2)
        c1.metric("Documents", stats["documents_total"])
        c2.metric("Chunks", stats["chunks"])
        if stats["documents"]:
            st.caption(" · ".join(f"{k}: {v}" for k, v in stats["documents"].items()))
        cache = stats["cache"]
        c1, c2 = st.columns(2)
        c1.metric("Cached answers", cache["semantic_entries"])
        c2.metric("Semantic hits", cache["semantic_hits"])
        with st.expander("Active configuration"):
            st.json(stats["config"])

    st.divider()
    b1, b2 = st.columns(2)
    if b1.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    if b2.button("Clear cache", width="stretch"):
        body, err = api("DELETE", "/cache")
        if err:
            st.error(err)
        else:
            st.toast(f"Removed {body['removed'] if body else 0} cache entries")
            get_stats.clear()
    if st.button("Ingest documents", width="stretch", help="Idempotent re-ingest"):
        with st.spinner("Ingesting data/docs ..."):
            body, err = api("POST", "/ingest", timeout=1800)
        if err:
            st.error(err)
        elif body:
            st.success(
                f"{body['docs_ingested']} ingested, {body['docs_skipped']} unchanged, "
                f"{body['chunks_written']} chunks in {body['seconds']}s"
            )
            get_stats.clear()


# ------------------------------------------------------------------------ rendering ----
def _sql_chart(df: pd.DataFrame) -> None:
    """Draw a chart when the result is one label/time column plus numeric columns."""
    if len(df) < 2 or len(df.columns) < 2 or len(df) > 200:
        return
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or len(labels) != 1:
        return
    label = labels[0]
    chart_df = df.copy()
    # The API returns timestamps as ISO strings; only those are treated as a time axis.
    is_time = bool(chart_df[label].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").all())
    if is_time:
        chart_df[label] = pd.to_datetime(chart_df[label], utc=True)
    chart_df = chart_df.set_index(label)[numeric[:3]]
    if is_time:
        st.line_chart(chart_df)
    else:
        st.bar_chart(chart_df)


def render_response(resp: dict[str, Any], *, debug: bool) -> None:
    if resp.get("refused"):
        st.warning(resp["answer"])
    else:
        st.markdown(resp["answer"])

    route = resp["route"]
    badges = [
        f":{ROUTE_COLORS.get(route['route'], 'gray')}-badge[route: {route['route']}]",
        f":gray-badge[router: {route['source']}]",
        f":{CACHE_COLORS.get(resp['cache'], 'gray')}-badge[cache: {resp['cache']}"
        + (f" {resp['cache_similarity']:.3f}" if resp.get("cache_similarity") else "")
        + "]",
        f":gray-badge[{resp['timings_ms'].get('total', 0) / 1000:.2f}s]",
    ]
    if resp.get("refused"):
        badges.append(f":red-badge[refused: {resp.get('refusal_reason') or 'no answer'}]")
    st.markdown(" ".join(badges))

    if resp.get("error"):
        st.error(resp["error"])

    sources = resp.get("sources") or []
    if sources:
        cited = [s for s in sources if s["cited"]]
        label = f"Sources ({len(cited)} cited of {len(sources)} retrieved)"
        with st.expander(label, expanded=bool(cited) and not resp.get("refused")):
            for s in sorted(sources, key=lambda s: (not s["cited"], s["n"])):
                mark = "📌" if s["cited"] else "▫️"
                st.markdown(
                    f"{mark} **[{s['n']}] {s['title']}**  \n"
                    f"`{s['key']}` · rerank score `{s['score']:.3f}`"
                )
                st.caption(s["snippet"].replace("\n", " ")[:500] + " …")

    sql = resp.get("sql")
    if sql:
        title = f"SQL · {sql['row_count']} rows" if sql["ok"] else "SQL · failed"
        if sql["attempts"] > 1:
            title += f" · {sql['attempts']} attempts"
        with st.expander(title, expanded=not sources):
            if sql.get("sql"):
                st.code(sql["sql"], language="sql")
            for e in sql.get("errors", []):
                st.caption(f"⚠️ attempt error: {e}")
            if sql["ok"] and sql["columns"]:
                df = pd.DataFrame(sql["rows"], columns=sql["columns"])
                st.dataframe(df, width="stretch", hide_index=True)
                _sql_chart(df)

    if debug and resp.get("timings_ms"):
        with st.expander("Timings & debug"):
            t = {k: v for k, v in resp["timings_ms"].items() if k != "total"}
            if t:
                st.bar_chart(pd.DataFrame({"ms": t}), horizontal=True)
            st.caption(f"Router reason: {route.get('reason') or '—'}")
            st.json(resp, expanded=False)


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    payload: dict[str, Any] = {"query": question, "use_cache": use_cache}
    if route_label != "Auto":
        payload["route"] = route_label.lower()
    if top_k != 5:
        payload["top_k"] = top_k
    with st.chat_message("assistant"), st.spinner("Thinking ..."):
        body, err = api("POST", "/query", json=payload)
    if err or body is None:
        st.session_state.messages.append({"role": "error", "content": err or "empty response"})
    else:
        st.session_state.messages.append({"role": "assistant", "response": body})
    get_stats.clear()


# ----------------------------------------------------------------------------- tabs ----
tab_chat, tab_retrieval, tab_about = st.tabs(["💬 Ask", "🧪 Retrieval inspector", "ℹ️ About"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            "Ask about internal **policies**, company **10-K filings**, the **employee FAQ**, "
            "or the **trading database**. The router picks the source automatically."
        )
        cols = st.columns(len(EXAMPLES))
        for col, (group, qs) in zip(cols, EXAMPLES.items(), strict=True):
            with col:
                st.markdown(f"**{group}**")
                for i, q in enumerate(qs):
                    if st.button(q, key=f"ex-{group}-{i}", width="stretch"):
                        st.session_state.pending = q
                        st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "error":
            st.chat_message("assistant").error(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_response(msg["response"], debug=show_debug)

    prompt = st.chat_input("Ask a question ...")
    question = prompt or st.session_state.pending
    if question:
        st.session_state.pending = None
        ask(question)
        st.rerun()

with tab_retrieval:
    st.markdown(
        "Inspect the retrieval stage directly: compare dense (pgvector HNSW), keyword "
        "(Postgres full-text) and hybrid (RRF) retrieval, with or without the cross-encoder."
    )
    with st.form("retrieve"):
        rq = st.text_input("Query", value="What is the hotel cap in the travel policy?")
        c1, c2, c3 = st.columns([2, 1, 1])
        modes = c1.multiselect(
            "Retrievers", ["hybrid", "vector", "keyword"], default=["hybrid", "vector", "keyword"]
        )
        rerank = c2.toggle("Cross-encoder rerank", value=True)
        k = c3.number_input("k", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Retrieve", type="primary")
    if submitted and rq and modes:
        cols = st.columns(len(modes))
        for col, mode in zip(cols, modes, strict=True):
            with col:
                st.markdown(f"#### {mode}")
                with st.spinner(f"{mode} ..."):
                    body, err = api(
                        "POST",
                        "/retrieve",
                        json={"query": rq, "mode": mode, "rerank": rerank, "k": int(k)},
                    )
                if err or body is None:
                    st.error(err)
                    continue
                st.caption(" · ".join(f"{s} {ms:.0f} ms" for s, ms in body["timings_ms"].items()))
                for s in body["sources"]:
                    st.markdown(f"**{s['n']}. {s['title']}**  \n`{s['key']}` · `{s['score']:.3f}`")
                    st.caption(s["snippet"].replace("\n", " ")[:300] + " …")

with tab_about:
    st.markdown(
        """
### How a question is answered

1. **Cache** — exact match in Redis, then a semantic match in Postgres (pgvector cosine
   similarity ≥ threshold, and identical numbers / codes so "2023" never serves "2024").
2. **Router** — LLM classifier (heuristic fallback) picks **docs**, **sql** or **hybrid**.
3. **Docs** — dense (nomic-embed-text, HNSW) and keyword (tsvector) retrieval run in
   parallel, fused with Reciprocal Rank Fusion, then a cross-encoder reranks the top 20 to 5.
   The LLM answers *only* from those sources and cites them as `[n]`; if the best rerank
   score is too low it refuses without calling the LLM.
4. **SQL** — schema-grounded few-shot Text2SQL with one self-repair attempt. Every query
   passes **SQLGuard** (single SELECT, allow-listed tables, no dangerous functions,
   enforced LIMIT) and runs as a read-only Postgres role with a statement timeout.
5. **Hybrid** — both paths run concurrently and the LLM combines them.

Everything runs locally: Postgres + pgvector, Redis, and Ollama.
"""
    )
