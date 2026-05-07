"""Streamlit frontend for the RAG Document Intelligence System.

Run:
    streamlit run frontend/app.py

Talks to the FastAPI backend at BACKEND_URL (default http://localhost:8000).
Make sure uvicorn is running: `uvicorn backend.main:app --reload`.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Make the project root importable when Streamlit runs this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
import streamlit as st

from backend.config import get_settings

logger = logging.getLogger(__name__)

SETTINGS = get_settings()
BACKEND = SETTINGS.backend_url.rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=10.0)


# ---------- Backend client ----------


def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND, timeout=TIMEOUT)


def _format_error(resp: httpx.Response) -> str:
    """Pull a human-readable message out of a structured ErrorResponse."""
    try:
        body = resp.json()
        return f"{body.get('status_code', resp.status_code)}: {body.get('message', resp.text)}"
    except Exception:
        return f"{resp.status_code}: {resp.text[:200]}"


def api_list_documents() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/api/documents")
    if r.status_code != 200:
        st.error(f"List failed — {_format_error(r)}")
        return []
    return r.json().get("documents", [])


def api_upload(filename: str, file_bytes: bytes, mime: str) -> dict[str, Any] | None:
    with _client() as c:
        r = c.post(
            "/api/documents/upload",
            files={"file": (filename, file_bytes, mime or "application/octet-stream")},
        )
    if r.status_code != 201:
        st.error(f"Upload failed — {_format_error(r)}")
        return None
    return r.json()


def api_delete(doc_id: str) -> bool:
    with _client() as c:
        r = c.delete(f"/api/documents/{doc_id}")
    if r.status_code != 200:
        st.error(f"Delete failed — {_format_error(r)}")
        return False
    return True


MAX_HISTORY_TURNS = 6  # last 6 turns (3 user + 3 assistant) replayed to the LLM


def api_query(
    question: str,
    doc_ids: list[str] | None,
    top_k: int,
    retrieval_only: bool = False,
    history: list[dict[str, Any]] | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "retrieval_only": retrieval_only,
    }
    if doc_ids:
        payload["doc_ids"] = list(doc_ids)
    if use_hybrid is not None:
        payload["use_hybrid"] = use_hybrid
    if use_reranker is not None:
        payload["use_reranker"] = use_reranker
    if history:
        payload["history"] = [
            {"role": h["role"], "content": h["content"]}
            for h in history[-MAX_HISTORY_TURNS:]
            if h.get("role") in ("user", "assistant") and h.get("content")
        ]
    with _client() as c:
        r = c.post("/api/query", json=payload)
    if r.status_code != 200:
        st.error(f"Query failed — {_format_error(r)}")
        return None
    return r.json()


def stream_query(
    question: str,
    doc_ids: list[str] | None,
    top_k: int,
    retrieval_only: bool,
    history: list[dict[str, Any]] | None,
    use_hybrid: bool | None,
    use_reranker: bool | None,
    sink: dict[str, Any],
) -> Iterator[str]:
    """Stream tokens from POST /api/query/stream.

    Yields plain-text deltas suitable for `st.write_stream`. Side effects:
    populates `sink["citations"]`, `sink["usage"]`, `sink["error"]` as
    non-token events arrive — read them once the generator is exhausted.
    """
    payload: dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "retrieval_only": retrieval_only,
    }
    if doc_ids:
        payload["doc_ids"] = list(doc_ids)
    if use_hybrid is not None:
        payload["use_hybrid"] = use_hybrid
    if use_reranker is not None:
        payload["use_reranker"] = use_reranker
    if history:
        payload["history"] = [
            {"role": h["role"], "content": h["content"]}
            for h in history[-MAX_HISTORY_TURNS:]
            if h.get("role") in ("user", "assistant") and h.get("content")
        ]

    sink.setdefault("citations", [])
    sink.setdefault("usage", {})
    sink.setdefault("error", None)

    cur_event: str | None = None
    with _client() as c:
        with c.stream("POST", "/api/query/stream", json=payload) as r:
            if r.status_code != 200:
                # Drain so we can show a useful error message.
                body = r.read().decode("utf-8", errors="replace")
                sink["error"] = f"{r.status_code}: {body[:200]}"
                return
            for raw in r.iter_lines():
                if not raw:
                    cur_event = None
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if line.startswith("event: "):
                    cur_event = line[len("event: ") :].strip()
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[len("data: ") :])
                    except json.JSONDecodeError:
                        continue
                    if cur_event == "token":
                        text = data.get("text") or ""
                        if text:
                            yield text
                    elif cur_event == "citations":
                        sink["citations"] = data.get("citations") or []
                    elif cur_event == "done":
                        sink["usage"] = {
                            "model": data.get("model", ""),
                            "input_tokens": int(data.get("input_tokens", 0)),
                            "output_tokens": int(data.get("output_tokens", 0)),
                        }
                    elif cur_event == "error":
                        sink["error"] = data.get("message") or "Unknown streaming error"


CONFIDENCE_THRESHOLD = 0.30  # cosine sim below this is flagged as low confidence
TEXT_PREVIEW_CHARS = 600


def _render_citations(
    citations: list[dict[str, Any]],
    show_text: bool,
) -> None:
    """Render a citations expander with optional chunk-text previews."""
    if not citations:
        return
    label = f"📎 {len(citations)} chunk(s)" if show_text else f"📎 {len(citations)} citation(s)"
    with st.expander(label):
        for c in citations:
            st.markdown(
                f"**[{c['rank']}]** `{c['filename']}` — page {c['page_number']}, "
                f"chunk {c['chunk_index']} · score `{c['score']:.4f}`"
            )
            if show_text and c.get("text"):
                preview = c["text"]
                if len(preview) > TEXT_PREVIEW_CHARS:
                    preview = preview[:TEXT_PREVIEW_CHARS] + "…"
                st.markdown(f"> {preview}")


def _maybe_warn_low_confidence(citations: list[dict[str, Any]]) -> None:
    """Show a yellow warning when the top similarity score looks weak."""
    if not citations:
        return
    top = max(c["score"] for c in citations)
    if top < CONFIDENCE_THRESHOLD:
        st.warning(
            f"⚠️ Low retrieval confidence (top score `{top:.3f}` < `{CONFIDENCE_THRESHOLD}`). "
            "The corpus may not cover this question — treat the answer with skepticism."
        )


def api_health_ok() -> bool:
    try:
        with _client() as c:
            r = c.get("/api/health")
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# ---------- Page setup ----------


st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Per-conversation history keyed by the sorted, joined doc_id list (or "__all__").
if "messages" not in st.session_state:
    st.session_state.messages = {}  # type: dict[str, list[dict]]
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []  # type: list[str]   # [] means "all documents"


def _scope_key(ids: list[str]) -> str:
    """Stable bucket key for chat history. Empty selection = '__all__'."""
    return ",".join(sorted(ids)) if ids else "__all__"


def _scope_label(ids: list[str], docs: list[dict[str, Any]]) -> str:
    """Human-readable scope label for the chat header."""
    if not ids:
        return "all documents"
    name_by_id = {d["doc_id"]: d["filename"] for d in docs}
    names = [name_by_id.get(i, i) for i in ids]
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:2])}, +{len(names) - 2} more"


# ---------- Sidebar ----------


with st.sidebar:
    st.title("📄 RAG Console")
    st.caption(f"Backend: `{BACKEND}`")

    if not api_health_ok():
        st.error("Backend unreachable. Start it with `uvicorn backend.main:app --reload`.")
        st.stop()

    # --- Upload ---
    st.subheader("Upload")
    upload = st.file_uploader(
        "Drop a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if upload is not None:
        # Avoid re-uploading the same file across reruns.
        sig = (upload.name, upload.size)
        if st.session_state.get("_last_upload_sig") != sig:
            with st.spinner(f"Ingesting {upload.name}…"):
                result = api_upload(upload.name, upload.getvalue(), upload.type or "")
            if result:
                st.success(
                    f"Ingested **{result['filename']}** "
                    f"({result['num_pages']} pages → {result['num_chunks']} chunks)."
                )
                if result.get("summary"):
                    st.info(f"📝 **TL;DR** — {result['summary']}")
                st.session_state["_last_upload_sig"] = sig
                st.rerun()

    st.divider()

    # --- Document selector ---
    st.subheader("Documents")
    docs = api_list_documents()

    if not docs:
        st.info("No documents yet. Upload one above.")
    else:
        valid_ids = {d["doc_id"] for d in docs}
        # Drop any stale ids from a previous selection (e.g. after a delete).
        prior = {i for i in st.session_state.selected_doc_ids if i in valid_ids}

        st.write("Scope query to (leave all unchecked for all documents):")

        # Quick "select all" / "clear" buttons for convenience.
        ba, bb = st.columns(2)
        if ba.button("Select all", use_container_width=True):
            for d in docs:
                st.session_state[f"docsel-{d['doc_id']}"] = True
            st.rerun()
        if bb.button("Clear", use_container_width=True):
            for d in docs:
                st.session_state[f"docsel-{d['doc_id']}"] = False
            st.rerun()

        # One checkbox per document. Initial state comes from prior selection;
        # subsequent renders read st.session_state[<key>] (set by the widget).
        selected: list[str] = []
        for d in docs:
            key = f"docsel-{d['doc_id']}"
            default = key not in st.session_state and d["doc_id"] in prior
            checked = st.checkbox(
                f"{d['filename']}  ·  {d['num_chunks']} chunks",
                value=default,
                key=key,
                help=d.get("summary") or None,
            )
            if checked:
                selected.append(d["doc_id"])
        st.session_state.selected_doc_ids = selected

        if selected:
            st.caption(f"Searching **{len(selected)}** of {len(docs)} document(s).")
        else:
            st.caption(f"Searching **all {len(docs)}** document(s).")

        # Per-doc delete buttons
        with st.expander("Manage"):
            for d in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"**{d['filename']}**")
                if d.get("summary"):
                    col1.caption(d["summary"])
                if col2.button("🗑", key=f"del-{d['doc_id']}", help="Delete this document"):
                    if api_delete(d["doc_id"]):
                        # Drop chat history for that doc, prune from selection,
                        # and clear its orphan checkbox state.
                        st.session_state.messages.pop(d["doc_id"], None)
                        st.session_state.selected_doc_ids = [
                            i for i in st.session_state.selected_doc_ids
                            if i != d["doc_id"]
                        ]
                        st.session_state.pop(f"docsel-{d['doc_id']}", None)
                        st.rerun()

    st.divider()

    # --- Query options ---
    st.subheader("Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=SETTINGS.top_k)
    retrieval_only = st.checkbox(
        "Retrieval only (skip LLM)",
        value=False,
        help="Return the matching chunks without calling the language model. Free and instant.",
    )
    show_chunks = st.checkbox(
        "Show chunk text in citations",
        value=False,
        help="Render the actual passage text under each citation instead of just metadata.",
    )
    use_hybrid = st.checkbox(
        "Hybrid search (BM25 + cosine)",
        value=SETTINGS.enable_hybrid_search,
        help="Fuse keyword (BM25) and semantic (cosine) rankings. Better recall on rare names/numbers.",
    )
    use_reranker = st.checkbox(
        "Cross-encoder rerank",
        value=SETTINGS.enable_reranker,
        help="Rerank candidates with a cross-encoder. Adds 50-200 ms per query but big quality bump.",
    )

    pipeline_bits = ["dense"]
    if use_hybrid:
        pipeline_bits.append("BM25")
        pipeline_bits.append("RRF")
    if use_reranker:
        pipeline_bits.append("rerank")
    st.caption("Retrieval: " + " → ".join(pipeline_bits))

    if st.button("Clear chat for this scope"):
        st.session_state.messages.pop(_scope_key(st.session_state.selected_doc_ids), None)
        st.rerun()

    st.caption(
        f"Provider: `{SETTINGS.llm_provider}` · "
        f"model: `{getattr(SETTINGS, f'{SETTINGS.llm_provider}_model', 'unknown')}`"
    )


# ---------- Main: chat ----------


# Resolve the active scope label and chat history bucket
scope_ids: list[str] = list(st.session_state.selected_doc_ids)
scope_key = _scope_key(scope_ids)
scope_label = _scope_label(scope_ids, docs)

st.markdown(f"### Ask a question — _{scope_label}_")

history: list[dict] = st.session_state.messages.setdefault(scope_key, [])

# Replay history. Whether to render chunk text follows the *current* sidebar
# toggle so users can flip it on retroactively without re-querying.
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        cits = msg.get("citations") or []
        if cits and msg["role"] == "assistant":
            _maybe_warn_low_confidence(cits)
            _render_citations(cits, show_text=show_chunks or msg.get("retrieval_only", False))
        if msg.get("usage"):
            u = msg["usage"]
            st.caption(f"_{u['model']} · in {u['input_tokens']} / out {u['output_tokens']}_")

# New question
placeholder = (
    "Retrieve chunks for…" if retrieval_only else "Ask, or attach a document via the 📎 button…"
)
# accept_file=True adds a paperclip icon for inline upload. The returned value is a
# dict-like ChatInputValue with .text (str) and ["files"] (list of UploadedFile).
chat_value = st.chat_input(
    placeholder,
    accept_file="multiple",
    file_type=["pdf", "docx", "txt"],
)

# 1) Handle any inline-uploaded files first, regardless of whether text was sent.
if chat_value and chat_value.get("files"):
    new_uploads: list[str] = []
    for f in chat_value["files"]:
        with st.spinner(f"Ingesting {f.name}…"):
            res = api_upload(f.name, f.getvalue(), f.type or "")
        if res:
            new_uploads.append(
                f"**{res['filename']}** ({res['num_pages']} pages → {res['num_chunks']} chunks)"
            )
    if new_uploads:
        st.success("Ingested: " + ", ".join(new_uploads))
        # Refresh the doc list so the new docs appear in the sidebar checkboxes.
        # Streamlit reruns automatically on next interaction; force one now if no
        # text follows, otherwise let the query path handle the rerun naturally.
        if not (chat_value.get("text") or "").strip():
            st.rerun()

# 2) If text was also submitted (or only text), run the query.
question = (chat_value.get("text") or "").strip() if chat_value else ""
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        prior_turns = history[:-1]

        if retrieval_only:
            # No tokens to stream; the sync endpoint is simpler and identical UX.
            with st.spinner("Retrieving…"):
                result = api_query(
                    question,
                    doc_ids=scope_ids or None,
                    top_k=top_k,
                    retrieval_only=True,
                    history=prior_turns,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                )
            if result is None:
                history.append({"role": "assistant", "content": "_(error — see above)_"})
            else:
                st.markdown(result["answer"])
                citations = result.get("citations", [])
                _maybe_warn_low_confidence(citations)
                _render_citations(citations, show_text=True)
                usage = {
                    "model": result["model"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                }
                st.caption(f"_{usage['model']} · in 0 / out 0_")
                history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": citations,
                        "usage": usage,
                        "retrieval_only": True,
                    }
                )
        else:
            sink: dict[str, Any] = {}
            stream_iter = stream_query(
                question,
                doc_ids=scope_ids or None,
                top_k=top_k,
                retrieval_only=False,
                history=prior_turns,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                sink=sink,
            )
            # st.write_stream renders incrementally and returns the concatenated text.
            answer_text = st.write_stream(stream_iter)

            if sink.get("error"):
                st.error(f"Streaming failed — {sink['error']}")
                history.append({"role": "assistant", "content": "_(error — see above)_"})
            else:
                citations = sink.get("citations", [])
                _maybe_warn_low_confidence(citations)
                _render_citations(citations, show_text=show_chunks)
                usage = sink.get("usage", {}) or {
                    "model": "(unknown)", "input_tokens": 0, "output_tokens": 0,
                }
                st.caption(
                    f"_{usage.get('model','?')} · "
                    f"in {usage.get('input_tokens',0)} / out {usage.get('output_tokens',0)}_"
                )
                history.append(
                    {
                        "role": "assistant",
                        "content": answer_text or "",
                        "citations": citations,
                        "usage": usage,
                        "retrieval_only": False,
                    }
                )
