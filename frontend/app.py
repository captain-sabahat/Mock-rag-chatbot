import streamlit as st
import requests
import base64
import os
from io import BytesIO
from typing import Optional, Dict, Any
from datetime import datetime

# =====================================================================================
# BASIC CONFIG
# =====================================================================================

st.set_page_config(
    page_title="RAG Chat Frontend",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default backend URL (must match FastAPI router prefix /api/v1)
DEFAULT_BACKEND_URL = "http://127.0.0.1:8001/api/v1"

# =====================================================================================
# SESSION STATE
# =====================================================================================

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

if "timeout" not in st.session_state:
    st.session_state.timeout = 30

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_default"

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_default"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "uploaded_file_meta" not in st.session_state:
    st.session_state.uploaded_file_meta = None


# =====================================================================================
# SIDEBAR
# =====================================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.session_state.backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
        help="FastAPI base URL (without /query suffix)",
    )
    st.session_state.timeout = st.number_input(
        "Timeout (seconds)", min_value=5, max_value=120, value=st.session_state.timeout
    )

    st.markdown("## 👤 Session")
    st.session_state.user_id = st.text_input(
        "User ID", value=st.session_state.user_id
    )
    st.session_state.session_id = st.text_input(
        "Session ID", value=st.session_state.session_id
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
    with col_sb2:
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.session_id = f"session_{int(datetime.now().timestamp())}"
            st.session_state.messages = []
            st.experimental_rerun()

    st.markdown("## ℹ️ Status")
    st.write(f"Messages: **{len(st.session_state.messages)}**")
    if st.session_state.uploaded_file_meta:
        st.write(
            f"Doc: `{st.session_state.uploaded_file_meta['file_name']}` "
            f"({st.session_state.uploaded_file_meta['file_type']}, "
            f"{st.session_state.uploaded_file_meta['file_size']//1024} KB)"
        )
    else:
        st.write("Doc: _none_")


# =====================================================================================
# HELPER: BUILD DOCUMENT ATTACHMENT
# =====================================================================================

def build_document_attachment(uploaded_file: Optional[BytesIO]) -> Optional[Dict[str, Any]]:
    """
    Build DocumentAttachment object to match backend schema:

    DocumentAttachment(BaseModel):
        file_name: str
        file_type: str  (pdf|docx|txt)
        file_size: int  (bytes, <= 50MB)
        file_content: Optional[str]  (base64)
    """
    if not uploaded_file:
        return None

    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    file_name = uploaded_file.name
    _, ext = os.path.splitext(file_name.lower())
    ext = ext.lstrip(".")

    # Map extension to allowed types in schema (pdf|docx|txt) [file:250]
    if ext not in {"pdf", "docx", "txt"}:
        # try to coerce md -> txt
        if ext == "md":
            ext = "txt"
        else:
            st.warning(f"Unsupported file type '{ext}'. Only pdf/docx/txt are allowed.")
            return None

    file_size = len(file_bytes)
    if file_size == 0:
        st.warning("Uploaded file is empty.")
        return None

    if file_size > 52_428_800:  # 50MB
        st.warning("Uploaded file exceeds 50MB limit.")
        return None

    encoded = base64.b64encode(file_bytes).decode("utf-8")

    return {
        "file_name": file_name,
        "file_type": ext,
        "file_size": file_size,
        "file_content": encoded,
    }


# =====================================================================================
# HELPER: SEND REQUEST TO /api/v1/query
# =====================================================================================

def send_user_query(
    prompt: str,
    document_attachment: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call backend /api/v1/query with EXACT UserQueryRequest schema. [file:250]

    UserQueryRequest(BaseModel):
        user_id: str
        session_id: str
        prompt: str
        document: Optional[DocumentAttachment]
        redis_lookup: RedisLookupFlag (\"yes\"|\"no\")
        doc_attached: DocAttachedFlag (\"yes\"|\"no\")
    """

    # Decide flags based on presence of document
    if document_attachment is None:
        doc_attached = "no"
    else:
        doc_attached = "yes"

    # For now, always allow redis lookup (you can expose this in UI later)
    redis_lookup = "yes"

    payload = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "prompt": prompt,
        "document": document_attachment,
        "redis_lookup": redis_lookup,
        "doc_attached": doc_attached,
    }  # [file:250]

    try:
        resp = requests.post(
            f"{st.session_state.backend_url}/query",
            json=payload,
            timeout=st.session_state.timeout,
        )
    except requests.Timeout:
        return {
            "status": "error",
            "error": "Request timeout",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }

    # For debugging 422, surface backend error body
    if resp.status_code != 200:
        try:
            data = resp.json()
        except Exception:
            data = {"detail": resp.text}
        return {
            "status": "error",
            "error": f"HTTP {resp.status_code}",
            "detail": data,
        }

    return resp.json()


# =====================================================================================
# MAIN UI
# =====================================================================================

st.title("💬 RAG Chat Frontend")

mode = st.segmented_control(
    "Select Mode",
    ["Query Only", "Query + Doc", "Summary"],
    default="Query Only",
)


# -------------------------------------------------------------------
# MODE 1: QUERY ONLY  (no document, doc_attached = \"no\")
# -------------------------------------------------------------------
if mode == "Query Only":
    st.subheader("💬 Ask a question")

    user_prompt = st.text_area(
        "Prompt",
        placeholder="Type your question...",
        height=140,
        key="prompt_query_only",
    )

    if st.button("Send", type="primary"):
        if not user_prompt or not user_prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            with st.spinner("Waiting for backend..."):
                result = send_user_query(
                    prompt=user_prompt.strip(),
                    document_attachment=None,
                )

            if result.get("status") == "error":
                st.error(result.get("error"))
                if "detail" in result:
                    st.json(result["detail"])
            else:
                answer = result.get("answer") or result.get("result") or ""
                st.markdown("### Answer")
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "user", "content": user_prompt.strip()}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


# -------------------------------------------------------------------
# MODE 2: QUERY + DOC  (document present, doc_attached = \"yes\")
# -------------------------------------------------------------------
elif mode == "Query + Doc":
    st.subheader("📄 Ask about a document")

    col_u, col_q = st.columns([1, 1.5])

    with col_u:
        upload = st.file_uploader(
            "Upload document (pdf/docx/txt/md)",
            type=["pdf", "docx", "txt", "md"],
            key="uploader_query_doc",
        )
        if upload is not None:
            st.session_state.uploaded_file = upload
            doc_meta = build_document_attachment(upload)
            st.session_state.uploaded_file_meta = doc_meta
            if doc_meta:
                st.success(
                    f"Loaded `{doc_meta['file_name']}` "
                    f"({doc_meta['file_type']}, "
                    f"{doc_meta['file_size']//1024} KB)"
                )

    with col_q:
        user_prompt = st.text_area(
            "Question about the document",
            placeholder="Type your question about the uploaded document...",
            height=160,
            key="prompt_query_doc",
        )

    if st.button("Send", type="primary"):
        if not user_prompt or not user_prompt.strip():
            st.error("Prompt cannot be empty.")
        elif not st.session_state.uploaded_file_meta:
            st.error("Please upload a document first.")
        else:
            with st.spinner("Waiting for backend..."):
                result = send_user_query(
                    prompt=user_prompt.strip(),
                    document_attachment=st.session_state.uploaded_file_meta,
                )

            if result.get("status") == "error":
                st.error(result.get("error"))
                if "detail" in result:
                    st.json(result["detail"])
            else:
                answer = result.get("answer") or result.get("result") or ""
                st.markdown("### Answer")
                st.write(answer)

                sources = result.get("sources") or []
                if sources:
                    st.markdown("### Sources")
                    for src in sources:
                        st.write(f"- {src}")

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": user_prompt.strip(),
                        "doc": st.session_state.uploaded_file_meta["file_name"],
                    }
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


# -------------------------------------------------------------------
# MODE 3: SUMMARY (document only, prompt is auto-generated)
# -------------------------------------------------------------------
elif mode == "Summary":
    st.subheader("📋 Summarize a document")

    upload = st.file_uploader(
        "Upload document (pdf/docx/txt/md)",
        type=["pdf", "docx", "txt", "md"],
        key="uploader_summary",
    )
    if upload is not None:
        st.session_state.uploaded_file = upload
        doc_meta = build_document_attachment(upload)
        st.session_state.uploaded_file_meta = doc_meta
        if doc_meta:
            st.success(
                f"Loaded `{doc_meta['file_name']}` "
                f"({doc_meta['file_type']}, "
                f"{doc_meta['file_size']//1024} KB)"
            )

    if st.button("Summarize", type="primary"):
        if not st.session_state.uploaded_file_meta:
            st.error("Please upload a document first.")
        else:
            # Backend requires non-empty prompt, so send a fixed summary prompt. [file:250]
            summary_prompt = "Provide a concise summary of the attached document."

            with st.spinner("Waiting for backend..."):
                result = send_user_query(
                    prompt=summary_prompt,
                    document_attachment=st.session_state.uploaded_file_meta,
                )

            if result.get("status") == "error":
                st.error(result.get("error"))
                if "detail" in result:
                    st.json(result["detail"])
            else:
                answer = result.get("answer") or result.get("result") or ""
                st.markdown("### Summary")
                st.write(answer)

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "[Document summary request]",
                        "doc": st.session_state.uploaded_file_meta["file_name"],
                    }
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


# =====================================================================================
# HISTORY
# =====================================================================================

st.markdown("---")
st.markdown("### Conversation History")

if not st.session_state.messages:
    st.info("No messages yet.")
else:
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")
