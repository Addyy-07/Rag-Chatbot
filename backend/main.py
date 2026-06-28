"""
backend/main.py
────────────────
Streamlit application entrypoint — UI logic ONLY.

This file's only job is to:
  1. Configure the Streamlit page.
  2. Cache expensive resources (embeddings) once per process.
  3. Render the UI and respond to user interactions.
  4. Delegate ALL business logic to routes (which call services).

What this file deliberately does NOT contain:
  - PDF parsing, text splitting, embedding, or LLM calls.
  - Direct Pinecone or Groq API calls.
  - CSS or HTML strings (imported from frontend/styles.py).
  - os.getenv() calls (config comes from settings).

Run:
    streamlit run backend/main.py
"""

import streamlit as st

from backend.config.settings import settings
from backend.routes.chat_router import handle_chat_query
from backend.routes.ingest_router import handle_pdf_upload
from backend.services.embedding_service import create_embedding_model
from backend.utils.file_utils import human_readable_size
from backend.utils.logger import configure_root_logger, get_logger
from frontend.styles import (
    FOOTER_HTML,
    GLOBAL_CSS,
    render_empty_chat_state,
    render_page_header,
    render_ready_badge,
    render_stat_card,
)

# ── Logging ────────────────────────────────────────────────────────────────────
configure_root_logger()
log = get_logger(__name__)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Cached resource: embeddings (loaded once per process) ─────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embeddings():
    """
    Load and cache the HuggingFace embeddings model.

    @st.cache_resource ensures this runs only once per Streamlit worker process,
    regardless of how many times the page reruns.
    """
    return create_embedding_model()


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown(
    render_page_header(
        f"{settings.app_icon} {settings.app_title}",
        "Upload a PDF and have a conversation with it",
    ),
    unsafe_allow_html=True,
)


# ── Upload Section ─────────────────────────────────────────────────────────────
if not st.session_state.get("pdf_ready"):

    uploaded_file = st.file_uploader(
        "📂 Choose a PDF file",
        type="pdf",
        help="Upload any PDF document to start chatting with it",
    )

    if uploaded_file:
        file_size = human_readable_size(uploaded_file.size)
        st.markdown(
            f"**📄 {uploaded_file.name}** &nbsp;·&nbsp; {file_size}"
        )

        if st.button("⚡ Process & Start Chatting", key="btn_process"):
            embeddings = _get_embeddings()
            with st.spinner("Reading and indexing your PDF — this may take a moment..."):
                try:
                    result = handle_pdf_upload(uploaded_file, embeddings)
                    st.session_state.pdf_ready = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_chunks = result.chunk_count
                    st.session_state.pdf_pages = result.page_count
                    st.session_state.messages = []
                    log.info(
                        "PDF '%s' processed: %d pages, %d chunks.",
                        uploaded_file.name,
                        result.page_count,
                        result.chunk_count,
                    )
                    st.rerun()
                except Exception as exc:
                    log.error("Ingestion failed: %s", exc, exc_info=True)
                    st.error(f"❌ Failed to process PDF: {exc}")


# ── Chat Section ───────────────────────────────────────────────────────────────
else:
    # ── Document info bar ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            render_ready_badge(st.session_state.pdf_name),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_stat_card(st.session_state.pdf_pages, "Pages"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_stat_card(st.session_state.pdf_chunks, "Chunks"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔄 Upload New PDF", key="btn_new_pdf"):
        st.session_state.pdf_ready = False
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Chat history initialisation ──────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Empty state ──────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(render_empty_chat_state(), unsafe_allow_html=True)

    # ── Render existing messages ─────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    embeddings = _get_embeddings()
                    answer = handle_chat_query(
                        user_input,
                        st.session_state.messages,
                        embeddings,
                    )
                except Exception as exc:
                    log.error("Chat query failed: %s", exc, exc_info=True)
                    answer = f"⚠️ Something went wrong: {exc}"

            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
