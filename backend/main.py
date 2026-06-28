"""
backend/main.py
────────────────
Streamlit application entrypoint — UI logic ONLY.

v2 — Multi-Document Support
  - Sidebar: document library (all ingested PDFs with metadata cards + delete)
  - Main panel: multi-file uploader
  - Chat mode selector: Single Doc | Selected Docs | All Docs
  - Namespace-aware chat pipeline

Layout:
  ┌─────────────────┬──────────────────────────────────────┐
  │   SIDEBAR       │          MAIN PANEL                  │
  │                 │                                      │
  │  📚 Library     │  [Upload Section]  OR  [Chat Section]│
  │  ─ doc card     │                                      │
  │  ─ doc card     │  Mode: ○ Single  ○ Selected  ○ All  │
  │  ─ [Delete]     │                                      │
  │                 │  [Chat interface]                    │
  └─────────────────┴──────────────────────────────────────┘

Run:
    streamlit run backend/main.py
"""

import streamlit as st
import httpx

from backend.config.settings import settings
from backend.routes.chat_router import handle_chat_query
from backend.routes.ingest_router import handle_pdf_upload
from backend.services.document_registry import DocumentRegistry
from backend.services.embedding_service import create_embedding_model
from backend.services import chat_api_client
from backend.services.usage_api_client import get_usage_limits, track_query
from backend.utils.file_utils import human_readable_size
from backend.utils.logger import configure_root_logger, get_logger
from backend.views.pricing_view import render_pricing_view
from backend.views.billing_view import render_billing_view
from frontend.styles import (
    FOOTER_HTML,
    GLOBAL_CSS,
    render_citation_cards,
    render_doc_count_badge,
    render_document_card,
    render_empty_chat_state,
    render_library_empty_state,
    render_mode_badge,
    render_page_header,
    render_sidebar_header,
    render_stat_card,
)

# ── Logging ────────────────────────────────────────────────────────────────────
configure_root_logger()
log = get_logger(__name__)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embeddings():
    """Load and cache the HuggingFace embeddings model (once per process)."""
    return create_embedding_model()


@st.cache_resource
def _get_registry() -> DocumentRegistry:
    """Return a cached DocumentRegistry singleton."""
    return DocumentRegistry()


# ── Sidebar — Document Library ────────────────────────────────────────────────

def _render_sidebar(registry: DocumentRegistry) -> None:
    """Render the document library in the sidebar."""
    with st.sidebar:
        st.markdown(
            render_page_header(f"{settings.app_icon} DocChat AI", f"v{settings.app_version}"),
            unsafe_allow_html=True,
        )
        user = st.session_state.get("user")
        if user:
            st.markdown(f"<div style='margin-bottom: 15px; color: #a5b4fc'>👤 Welcome, {user.get('full_name') or user.get('username')}</div>", unsafe_allow_html=True)
            
        st.markdown(render_sidebar_header("📚 DOCUMENT LIBRARY"), unsafe_allow_html=True)

        docs = registry.get_all(owner_id=user.get("id") if user else None)

        if not docs:
            st.markdown(render_library_empty_state(), unsafe_allow_html=True)
        else:
            st.markdown(
                render_doc_count_badge(len(docs)),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            for doc in docs:
                st.markdown(render_document_card(doc), unsafe_allow_html=True)

                col_del, col_space = st.columns([1, 2])
                with col_del:
                    if st.button(
                        "🗑️ Delete",
                        key=f"del_{doc.document_id}",
                        help=f"Remove '{doc.filename}' from the library",
                    ):
                        embeddings = _get_embeddings()
                        deleted = registry.delete(doc.document_id, embeddings)
                        if deleted:
                            # Clear any chat scoped to this doc
                            if st.session_state.get("selected_doc_ids"):
                                st.session_state.selected_doc_ids = [
                                    d for d in st.session_state.selected_doc_ids
                                    if d != doc.document_id
                                ]
                            st.session_state.messages = []
                            st.success(f"✅ '{doc.display_name}' deleted.")
                            st.rerun()

                st.markdown("<hr style='border-color:#1e2130;margin:4px 0 10px'>", unsafe_allow_html=True)
                
        if st.session_state.get("jwt_token"):
            st.markdown(render_sidebar_header("💬 CHAT HISTORY"), unsafe_allow_html=True)
            token = st.session_state.get("jwt_token")
            sessions = chat_api_client.get_chat_sessions(token)
            
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.pop("active_chat_id", None)
                st.session_state.pop("loaded_chat_id", None)
                st.session_state.messages = []
                st.rerun()
                
            for sess in sessions:
                col_chat_btn, col_chat_del = st.columns([4, 1])
                with col_chat_btn:
                    is_active = (st.session_state.get("active_chat_id") == sess["id"])
                    label = f"💬 **{sess['title']}**" if is_active else f"💬 {sess['title']}"
                    if st.button(label, key=f"chat_btn_{sess['id']}", use_container_width=True):
                        st.session_state.active_chat_id = sess["id"]
                        st.rerun()
                with col_chat_del:
                    if st.button("🗑️", key=f"chat_del_{sess['id']}", help="Delete chat"):
                        chat_api_client.delete_chat_session(sess["id"], token)
                        if st.session_state.get("active_chat_id") == sess["id"]:
                            st.session_state.pop("active_chat_id", None)
                            st.session_state.pop("loaded_chat_id", None)
                            st.session_state.messages = []
                        st.rerun()
                        
            # ── Main Navigation ────────────────────────────────────────────────────
            st.markdown("### 🧭 Navigation")
            if "current_page" not in st.session_state:
                st.session_state.current_page = "chat"
                
            col_nav1, col_nav2, col_nav3 = st.columns(3)
            with col_nav1:
                if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.current_page == "chat" else "secondary"):
                    st.session_state.current_page = "chat"
                    st.rerun()
            with col_nav2:
                if st.button("🚀 Pricing", use_container_width=True, type="primary" if st.session_state.current_page == "pricing" else "secondary"):
                    st.session_state.current_page = "pricing"
                    st.rerun()
            with col_nav3:
                if st.button("💳 Billing", use_container_width=True, type="primary" if st.session_state.current_page == "billing" else "secondary"):
                    st.session_state.current_page = "billing"
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ── Usage Limits UI ────────────────────────────────────────────────────
            usage = get_usage_limits(token)
            if usage:
                st.markdown("### 📊 SaaS Usage")
                tier_name = usage.get("tier", "free").upper()
                st.caption(f"**Current Plan:** {tier_name}")
                
                # PDFs
                c_pdfs = usage.get("current_pdfs", 0)
                m_pdfs = usage.get("max_pdfs", 5)
                pdf_pct = min(100, int((c_pdfs / m_pdfs) * 100)) if m_pdfs < 999999 else 0
                if m_pdfs < 999999:
                    st.progress(pdf_pct, text=f"PDFs: {c_pdfs} / {m_pdfs}")
                else:
                    st.caption(f"PDFs: {c_pdfs} / ∞")
                    
                # Queries
                c_queries = usage.get("current_queries", 0)
                m_queries = usage.get("max_queries", 50)
                q_pct = min(100, int((c_queries / m_queries) * 100)) if m_queries < 999999 else 0
                if m_queries < 999999:
                    st.progress(q_pct, text=f"Queries today: {c_queries} / {m_queries}")
                else:
                    st.caption(f"Queries today: {c_queries} / ∞")
                    
                if tier_name == "FREE":
                    st.info("Upgrade to Pro for unlimited usage! 🚀")
            
            st.markdown("---")
            if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        # ── Page Routing ───────────────────────────────────────────────────────────
        if st.session_state.get("current_page") == "pricing":
            render_pricing_view(token)
            return
        elif st.session_state.get("current_page") == "billing":
            render_billing_view(token)
            return

    # ── Main Chat Window ─────────────────────────────────────────────────────────────

def _render_upload_section(registry: DocumentRegistry) -> None:
    """Render the multi-file upload panel."""
    st.markdown(
        render_page_header(
            f"{settings.app_icon} {settings.app_title}",
            "Upload PDFs and have intelligent conversations with your documents",
        ),
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        f"📂 Upload up to {settings.max_upload_files} PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Hold Ctrl/Cmd to select multiple files",
        key="file_uploader",
    )

    if uploaded_files:
        # Cap at max_upload_files
        if len(uploaded_files) > settings.max_upload_files:
            st.warning(
                f"⚠️ Maximum {settings.max_upload_files} files allowed. "
                f"Only the first {settings.max_upload_files} will be processed."
            )
            uploaded_files = uploaded_files[: settings.max_upload_files]

        # Preview list
        st.markdown("**Selected files:**")
        for f in uploaded_files:
            st.markdown(
                f"&nbsp;&nbsp;📄 **{f.name}** &nbsp;·&nbsp; {human_readable_size(f.size)}"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            f"⚡ Process {len(uploaded_files)} PDF{'s' if len(uploaded_files) > 1 else ''}",
            key="btn_process",
        ):
            embeddings = _get_embeddings()
            progress_bar = st.progress(0, text="Starting ingestion...")

            try:
                results = []
                for i, uf in enumerate(uploaded_files):
                    progress_bar.progress(
                        (i) / len(uploaded_files),
                        text=f"Processing {uf.name} ({i + 1}/{len(uploaded_files)})...",
                    )
                    user_id = st.session_state.get("user", {}).get("id") if st.session_state.get("user") else None
                    token = st.session_state.get("jwt_token")
                    batch = handle_pdf_upload([uf], embeddings, registry, token=token, owner_id=user_id)
                    results.extend(batch)

                progress_bar.progress(1.0, text="All done!")

                if results:
                    st.session_state.messages = []
                    names = ", ".join(f"'{r.filename}'" for r in results)
                    st.success(
                        f"✅ Successfully ingested {len(results)} document(s): {names}"
                    )
                    log.info(
                        "Upload complete: %d/%d files ingested.",
                        len(results), len(uploaded_files),
                    )
                    st.rerun()
                else:
                    st.error("❌ No files were successfully processed. Check logs.")

            except Exception as exc:
                log.error("Batch ingestion failed: %s", exc, exc_info=True)
                st.error(f"❌ Ingestion error: {exc}")


# ── Chat Section ───────────────────────────────────────────────────────────────

def _render_chat_section(registry: DocumentRegistry) -> None:
    """Render the multi-mode chat interface."""
    user = st.session_state.get("user")
    docs = registry.get_all(owner_id=user.get("id") if user else None)

    if not docs:
        st.info("📭 Your library is empty. Upload PDFs using the sidebar.")
        return

    st.markdown(
        render_page_header(
            f"{settings.app_icon} {settings.app_title}",
            "Chat with your document library",
        ),
        unsafe_allow_html=True,
    )

    active_chat_id = st.session_state.get("active_chat_id")
    token = st.session_state.get("jwt_token")
    
    # ── Load existing session ───────────────────────────────────────────────────
    if active_chat_id:
        if "loaded_chat_id" not in st.session_state or st.session_state.loaded_chat_id != active_chat_id:
            sess_detail = chat_api_client.get_chat_session(active_chat_id, token)
            if sess_detail:
                st.session_state.messages = sess_detail["messages"]
                st.session_state.target_namespaces = sess_detail["target_namespaces"]
                st.session_state.loaded_chat_id = active_chat_id
                st.session_state.chat_title = sess_detail["title"]
                
        target_namespaces = st.session_state.get("target_namespaces", [])
        st.markdown(f"### {st.session_state.get('chat_title', 'Chat')}")
        st.caption(f"Querying **{len(target_namespaces)}** namespace(s)")
        
        col_rename, _ = st.columns([2, 3])
        with col_rename:
            new_title = st.text_input("Rename chat", value=st.session_state.get('chat_title', ''), label_visibility="collapsed")
            if new_title and new_title != st.session_state.get('chat_title', ''):
                if chat_api_client.rename_chat_session(active_chat_id, new_title, token):
                    st.session_state.chat_title = new_title
                    st.rerun()
    else:
        # ── Mode selector ──────────────────────────────────────────────────────────
        st.markdown("<div class='mode-header'>CHAT MODE</div>", unsafe_allow_html=True)
        mode = st.radio(
            "Chat mode",
            options=["📄 Single Document", "📚 Select Documents", "🌐 All Documents"],
            horizontal=True,
            label_visibility="collapsed",
            key="chat_mode",
        )
    
        # ── Namespace resolution based on mode ─────────────────────────────────────
        target_namespaces = []
        mode_key = "single"
    
        if mode == "📄 Single Document":
            mode_key = "single"
            doc_options = {d.filename: d.document_id for d in docs}
            selected_name = st.selectbox(
                "Select document",
                options=list(doc_options.keys()),
                key="single_doc_select",
            )
            if selected_name:
                target_namespaces = [doc_options[selected_name]]
    
        elif mode == "📚 Select Documents":
            mode_key = "selected"
            doc_options = {d.filename: d.document_id for d in docs}
            selected_names = st.multiselect(
                "Select documents to include",
                options=list(doc_options.keys()),
                default=list(doc_options.keys())[:2] if len(doc_options) >= 2 else list(doc_options.keys()),
                key="multi_doc_select",
            )
            target_namespaces = [doc_options[n] for n in selected_names if n in doc_options]
            if not target_namespaces:
                st.warning("⚠️ Please select at least one document.")
    
        else:  # All Documents
            mode_key = "all"
            target_namespaces = registry.get_all_namespaces(owner_id=user.get("id") if user else None)
            st.caption(f"🌐 Searching across **{len(target_namespaces)}** document(s)")
    
        # ── Mode badge + context info ──────────────────────────────────────────────
        if target_namespaces:
            col_badge, col_info = st.columns([2, 3])
            with col_badge:
                st.markdown(render_mode_badge(mode_key), unsafe_allow_html=True)
            with col_info:
                ns_count = len(target_namespaces)
                st.caption(
                    f"Querying **{ns_count}** namespace{'s' if ns_count > 1 else ''}"
                )
    
        # Clear messages when mode or selection changes (only if no active chat)
        mode_state_key = f"{mode_key}_{','.join(sorted(target_namespaces))}"
        if st.session_state.get("_last_mode_key") != mode_state_key:
            st.session_state.messages = []
            st.session_state["_last_mode_key"] = mode_state_key

    st.divider()

    # ── Chat history ───────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []


    col_chat, col_actions = st.columns([5, 1])
    with col_actions:
        if st.button("🔄 Clear Chat", key="btn_clear"):
            st.session_state.messages = []
            st.rerun()

    # ── Empty state ────────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(render_empty_chat_state(), unsafe_allow_html=True)

    # ── Render conversation ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Show citations stored with this message
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander(f"📎 {len(msg['citations'])} source(s)", expanded=False):
                    st.markdown(
                        render_citation_cards(msg["citations"]),
                        unsafe_allow_html=True,
                    )

    # ── Chat input ─────────────────────────────────────────────────────────────
    if not target_namespaces:
        st.chat_input("Select documents above to start chatting...", disabled=True)
        return

    if user_input := st.chat_input("Ask a question about your document(s)..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # ── Enforce SaaS Limit ─────────────────────────────────────────
                success, error_msg = track_query(token)
                if not success:
                    st.error(error_msg)
                    st.stop()
                    
                try:
                    embeddings = _get_embeddings()
                    result = handle_chat_query(
                        question=user_input,
                        history=st.session_state.messages,
                        embeddings=embeddings,
                        namespaces=target_namespaces,
                    )
                    answer = result.answer
                    citations = result.citations
                except ValueError as exc:
                    answer = f"⚠️ {exc}"
                    citations = []
                except Exception as exc:
                    log.error("Chat query failed: %s", exc, exc_info=True)
                    answer = f"⚠️ Something went wrong: {exc}"
                    citations = []

            st.write(answer)

            # Render citations inline — visible immediately after the answer
            if citations:
                with st.expander(f"📎 {len(citations)} source(s)", expanded=True):
                    st.markdown(
                        render_citation_cards(citations),
                        unsafe_allow_html=True,
                    )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )
        
        # ── Save to database ───────────────────────────────────────────────────
        if not active_chat_id:
            title = user_input[:30] + "..." if len(user_input) > 30 else user_input
            new_id = chat_api_client.create_chat_session(title, target_namespaces, token)
            if new_id:
                active_chat_id = new_id
                st.session_state.active_chat_id = new_id
                st.session_state.loaded_chat_id = new_id
                st.session_state.chat_title = title
                
        if active_chat_id:
            chat_api_client.add_message(active_chat_id, "user", user_input, [], token)
            # handle citations namedtuple structure safely
            citations_safe = []
            if citations:
                for c in citations:
                    # c is a Citation namedtuple: c.source_file, c.page_number, c.chunk_text
                    citations_safe.append({
                        "source_file": getattr(c, "source_file", ""),
                        "page_number": getattr(c, "page_number", None),
                        "chunk_text": getattr(c, "chunk_text", "")
                    })
            chat_api_client.add_message(active_chat_id, "assistant", answer, citations_safe, token)

# ── Auth Gate ──────────────────────────────────────────────────────────────────

def _render_auth_page():
    """Render login and signup tabs."""
    st.markdown(
        render_page_header(f"{settings.app_icon} {settings.app_title}", "Login to access your documents"),
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if not email or not password:
                        st.error("Please enter email and password.")
                    else:
                        try:
                            resp = httpx.post(
                                f"{settings.api_base_url}/auth/login",
                                json={"email": email, "password": password}
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.jwt_token = data["access_token"]
                                st.session_state.user = data["user"]
                                st.success("Logged in successfully!")
                                st.rerun()
                            else:
                                st.error(f"Login failed: {resp.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
                            
        with tab_signup:
            with st.form("signup_form"):
                s_email = st.text_input("Email")
                s_username = st.text_input("Username")
                s_name = st.text_input("Full Name (Optional)")
                s_password = st.text_input("Password", type="password")
                s_submit = st.form_submit_button("Sign Up", use_container_width=True)
                
                if s_submit:
                    if not s_email or not s_username or not s_password:
                        st.error("Please fill all required fields.")
                    elif len(s_password) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        try:
                            resp = httpx.post(
                                f"{settings.api_base_url}/auth/signup",
                                json={
                                    "email": s_email,
                                    "username": s_username,
                                    "password": s_password,
                                    "full_name": s_name if s_name else None
                                }
                            )
                            if resp.status_code == 201:
                                data = resp.json()
                                st.session_state.jwt_token = data["access_token"]
                                st.session_state.user = data["user"]
                                st.success("Account created successfully!")
                                st.rerun()
                            else:
                                st.error(f"Signup failed: {resp.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")

# ── App entrypoint ─────────────────────────────────────────────────────────────

registry = _get_registry()

# Check authentication
if not st.session_state.get("jwt_token"):
    _render_auth_page()
else:
    # Render sidebar (always visible)
    _render_sidebar(registry)

    # Render main panel based on library state
    user = st.session_state.get("user")
    docs = registry.get_all(owner_id=user.get("id") if user else None)
    if not docs:
        _render_upload_section(registry)
    else:
        # Show tabs: Chat | Upload More
        tab_chat, tab_upload = st.tabs(["💬 Chat", "📤 Upload More PDFs"])
        with tab_chat:
            _render_chat_section(registry)
        with tab_upload:
            _render_upload_section(registry)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
