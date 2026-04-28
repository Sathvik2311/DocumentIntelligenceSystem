"""Streamlit frontend for the RAG Document Intelligence System.

Run:
    streamlit run frontend/app.py

Talks to the FastAPI backend at BACKEND_URL (default http://localhost:8000).
Make sure uvicorn is running: `uvicorn backend.main:app --reload`.
"""

from __future__ import annotations

import logging
import sys
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


def api_query(
    question: str, doc_id: str | None, top_k: int
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"question": question, "top_k": top_k}
    if doc_id:
        payload["doc_id"] = doc_id
    with _client() as c:
        r = c.post("/api/query", json=payload)
    if r.status_code != 200:
        st.error(f"Query failed — {_format_error(r)}")
        return None
    return r.json()


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

# Per-conversation history keyed by selected doc_id (or "__all__").
if "messages" not in st.session_state:
    st.session_state.messages = {}  # type: dict[str, list[dict]]
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None  # None means "all documents"


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
                st.session_state["_last_upload_sig"] = sig
                st.rerun()

    st.divider()

    # --- Document selector ---
    st.subheader("Documents")
    docs = api_list_documents()

    if not docs:
        st.info("No documents yet. Upload one above.")
    else:
        labels = ["All documents"] + [
            f"{d['filename']}  ·  {d['num_chunks']} chunks" for d in docs
        ]
        ids = [None] + [d["doc_id"] for d in docs]

        # Preserve current selection across reruns when possible.
        try:
            current_idx = ids.index(st.session_state.selected_doc_id)
        except ValueError:
            current_idx = 0

        choice = st.radio(
            "Scope query to:",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            index=current_idx,
            label_visibility="collapsed",
        )
        st.session_state.selected_doc_id = ids[choice]

        # Per-doc delete buttons
        with st.expander("Manage"):
            for d in docs:
                col1, col2 = st.columns([4, 1])
                col1.write(f"**{d['filename']}**")
                if col2.button("🗑", key=f"del-{d['doc_id']}", help="Delete this document"):
                    if api_delete(d["doc_id"]):
                        # Drop chat history for that doc, reset selection if needed
                        st.session_state.messages.pop(d["doc_id"], None)
                        if st.session_state.selected_doc_id == d["doc_id"]:
                            st.session_state.selected_doc_id = None
                        st.rerun()

    st.divider()

    # --- Query options ---
    st.subheader("Settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=SETTINGS.top_k)

    if st.button("Clear chat for this scope"):
        key = st.session_state.selected_doc_id or "__all__"
        st.session_state.messages.pop(key, None)
        st.rerun()

    st.caption(
        f"Provider: `{SETTINGS.llm_provider}` · "
        f"model: `{getattr(SETTINGS, f'{SETTINGS.llm_provider}_model', 'unknown')}`"
    )


# ---------- Main: chat ----------


# Resolve the active scope label and chat history bucket
scope_id = st.session_state.selected_doc_id
scope_key = scope_id or "__all__"
scope_label = "all documents"
if scope_id:
    match = next((d for d in docs if d["doc_id"] == scope_id), None)
    scope_label = match["filename"] if match else scope_id

st.markdown(f"### Ask a question — _{scope_label}_")

history: list[dict] = st.session_state.messages.setdefault(scope_key, [])

# Replay history
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} citation(s)"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**[{c['rank']}]** `{c['filename']}` — page {c['page_number']}, "
                        f"chunk {c['chunk_index']} · score `{c['score']:.4f}`"
                    )
        if msg.get("usage"):
            u = msg["usage"]
            st.caption(f"_{u['model']} · in {u['input_tokens']} / out {u['output_tokens']}_")

# New question
question = st.chat_input("Ask anything about the selected document…", disabled=not docs)
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = api_query(question, doc_id=scope_id, top_k=top_k)

        if result is None:
            # api_query already surfaced the error
            history.append({"role": "assistant", "content": "_(error — see above)_"})
        else:
            st.markdown(result["answer"])
            citations = result.get("citations", [])
            if citations:
                with st.expander(f"📎 {len(citations)} citation(s)"):
                    for c in citations:
                        st.markdown(
                            f"**[{c['rank']}]** `{c['filename']}` — page {c['page_number']}, "
                            f"chunk {c['chunk_index']} · score `{c['score']:.4f}`"
                        )
            usage = {
                "model": result["model"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
            st.caption(f"_{usage['model']} · in {usage['input_tokens']} / out {usage['output_tokens']}_")
            history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": citations,
                    "usage": usage,
                }
            )
