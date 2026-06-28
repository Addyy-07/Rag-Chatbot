"""
frontend/styles.py
───────────────────
All CSS and HTML string constants / component builders for the Streamlit UI.

v2 additions:
  - Document library card component
  - Chat mode selector styling
  - Multi-column document grid CSS
  - Mode badge renderers
  - Library empty state
"""

# ── Global page CSS ────────────────────────────────────────────────────────────
GLOBAL_CSS: str = """
<style>
    /* ── Hide Streamlit chrome ─────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}

    /* ── Page background ───────────────────────────────────────── */
    .stApp { background-color: #0f1117; }

    /* ── Typography ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero header ────────────────────────────────────────────── */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #6b7280;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 28px;
    }

    /* ── Stat cards ─────────────────────────────────────────────── */
    .stat-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
        border: 1px solid #2e3047;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .stat-card:hover { border-color: #667eea; }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Primary button ─────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.2px;
        transition: opacity 0.2s ease, transform 0.1s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Danger button (delete) ──────────────────────────────────── */
    .danger-btn > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    }

    /* ── Chat messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
        border-radius: 14px;
        border: 1px solid #2e3047;
        margin-bottom: 10px;
        padding: 4px 8px;
    }

    /* ── Document library card ──────────────────────────────────── */
    .doc-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
        border: 1px solid #2e3047;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease, transform 0.15s ease;
        position: relative;
    }
    .doc-card:hover {
        border-color: #667eea;
        transform: translateY(-1px);
    }
    .doc-card-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 4px;
        word-break: break-word;
    }
    .doc-card-meta {
        font-size: 0.72rem;
        color: #6b7280;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 6px;
    }
    .doc-card-meta span {
        display: flex;
        align-items: center;
        gap: 3px;
    }
    .doc-card-id {
        font-size: 0.65rem;
        color: #374151;
        margin-top: 6px;
        font-family: monospace;
    }

    /* ── Mode selector ───────────────────────────────────────────── */
    .mode-header {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .mode-badge-single {
        background-color: #1a2744;
        color: #60a5fa;
        border: 1px solid #3b82f6;
    }
    .mode-badge-selected {
        background-color: #1e2a1a;
        color: #34d399;
        border: 1px solid #10b981;
    }
    .mode-badge-all {
        background-color: #2a1e2a;
        color: #c084fc;
        border: 1px solid #a855f7;
    }

    /* ── Status badges ──────────────────────────────────────────── */
    .ready-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #0d2818;
        color: #34d399;
        border: 1px solid #34d399;
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .doc-count-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #1a1d27, #1e2130);
        color: #667eea;
        border: 1px solid #667eea;
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* ── Upload area enhancement ────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2e3047;
        border-radius: 14px;
        padding: 8px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #667eea; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #13151f 100%);
        border-right: 1px solid #1e2130;
    }
    .sidebar-header {
        font-size: 0.72rem;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid #1e2130;
        margin-bottom: 10px;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .powered-by {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        margin-top: 32px;
        padding-bottom: 20px;
        letter-spacing: 0.3px;
    }
    .powered-by span { color: #4b5563; font-weight: 600; }

    /* ── Empty states ─────────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 50px 20px;
        animation: fadeIn 0.4s ease;
    }
    .empty-state-icon { font-size: 3rem; margin-bottom: 12px; }
    .empty-state-text { color: #6b7280; font-size: 0.95rem; }

    .library-empty {
        text-align: center;
        padding: 30px 16px;
        color: #4b5563;
        font-size: 0.85rem;
    }

    /* ── Progress / success ─────────────────────────────────────── */
    .success-banner {
        background: linear-gradient(135deg, #0d2818, #0a2010);
        border: 1px solid #34d399;
        border-radius: 12px;
        padding: 12px 16px;
        color: #34d399;
        font-size: 0.88rem;
        font-weight: 500;
        margin: 8px 0;
    }

    /* ── Animations ─────────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-8px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    .doc-card { animation: slideIn 0.25s ease; }
</style>
"""

# ── HTML component builders ────────────────────────────────────────────────────

def render_page_header(title: str, subtitle: str) -> str:
    """Return the HTML for the hero title + subtitle block."""
    return (
        f'<p class="main-title">{title}</p>'
        f'<p class="sub-title">{subtitle}</p>'
    )


def render_stat_card(value: str | int, label: str) -> str:
    """Return HTML for a single stat card."""
    return (
        f'<div class="stat-card">'
        f'<div class="stat-number">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>"
    )


def render_ready_badge(pdf_name: str) -> str:
    """Return the green 'ready' badge showing active PDF name."""
    return f'<span class="ready-badge">✅ {pdf_name}</span>'


def render_doc_count_badge(count: int) -> str:
    """Return a purple library count badge."""
    label = "document" if count == 1 else "documents"
    return f'<span class="doc-count-badge">📚 {count} {label} in library</span>'


def render_document_card(doc) -> str:
    """
    Return HTML for a document library card.

    Args:
        doc: A DocumentRecord instance.
    """
    from backend.utils.file_utils import human_readable_size
    size_str = human_readable_size(doc.size_bytes) if doc.size_bytes else "—"
    short_id = doc.document_id[:8]
    return (
        f'<div class="doc-card">'
        f'  <div class="doc-card-name">📄 {doc.display_name}</div>'
        f'  <div class="doc-card-meta">'
        f'    <span>📅 {doc.upload_date_display}</span>'
        f'    <span>📑 {doc.page_count} pages</span>'
        f'    <span>🧩 {doc.chunk_count} chunks</span>'
        f'    <span>💾 {size_str}</span>'
        f'  </div>'
        f'  <div class="doc-card-id">ID: {short_id}...</div>'
        f"</div>"
    )


def render_mode_badge(mode: str) -> str:
    """
    Return a mode indicator badge HTML.

    Args:
        mode: "single" | "selected" | "all"
    """
    badges = {
        "single":   ('<span class="mode-badge mode-badge-single">📄 Single Doc</span>'),
        "selected": ('<span class="mode-badge mode-badge-selected">📚 Selected Docs</span>'),
        "all":      ('<span class="mode-badge mode-badge-all">🌐 All Docs</span>'),
    }
    return badges.get(mode, "")


def render_empty_chat_state() -> str:
    """Return HTML for the empty chat placeholder."""
    return (
        '<div class="empty-state">'
        '<div class="empty-state-icon">💬</div>'
        '<div class="empty-state-text">Ask anything about your selected document(s) below</div>'
        "</div>"
    )


def render_library_empty_state() -> str:
    """Return HTML for the empty document library state."""
    return (
        '<div class="library-empty">'
        "📭 No documents uploaded yet.<br>"
        "Upload PDFs using the panel on the right."
        "</div>"
    )


def render_sidebar_header(text: str) -> str:
    """Return a styled sidebar section header."""
    return f'<div class="sidebar-header">{text}</div>'


FOOTER_HTML: str = """
<div class="powered-by">
    🤗 <span>HuggingFace</span> &nbsp;·&nbsp;
    📦 <span>Pinecone</span> &nbsp;·&nbsp;
    ⚡ <span>Groq Llama 3</span> &nbsp;·&nbsp;
    🦜 <span>LangChain</span>
</div>
"""
