"""
frontend/styles.py
───────────────────
All CSS and HTML string constants for the Streamlit UI.

Why extract styles here?
  - main.py becomes pure logic — no mixed HTML strings.
  - Designers can update styles without reading Python business logic.
  - Strings are typed constants: easy to test, reference, and lint.
  - Prepares for a future React/Next.js frontend migration where these
    become actual CSS files.

Usage
-----
    from frontend.styles import GLOBAL_CSS, render_stat_card

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(render_stat_card("42", "Chunks"), unsafe_allow_html=True)
"""

# ── Global page CSS ────────────────────────────────────────────────────────────
GLOBAL_CSS: str = """
<style>
    /* ── Hide Streamlit chrome ─────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    /* ── Page background ───────────────────────────────────────── */
    .stApp { background-color: #0f1117; }

    /* ── Typography ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
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
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.2px;
        transition: opacity 0.2s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ── Chat messages ──────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2130 100%);
        border-radius: 14px;
        border: 1px solid #2e3047;
        margin-bottom: 10px;
        padding: 4px 8px;
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

    /* ── Upload area enhancement ────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2e3047;
        border-radius: 14px;
        padding: 8px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #667eea; }

    /* ── Footer ─────────────────────────────────────────────────── */
    .powered-by {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        margin-top: 32px;
        padding-bottom: 20px;
        letter-spacing: 0.3px;
    }
    .powered-by span {
        color: #4b5563;
        font-weight: 600;
    }

    /* ── Empty state ─────────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 50px 20px;
        animation: fadeIn 0.4s ease;
    }
    .empty-state-icon { font-size: 3rem; margin-bottom: 12px; }
    .empty-state-text { color: #6b7280; font-size: 0.95rem; }

    /* ── Animations ─────────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
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
    """
    Return the HTML for a single stat card.

    Args:
        value: The large number/text to display.
        label: The small label beneath the value.
    """
    return (
        f'<div class="stat-card">'
        f'<div class="stat-number">{value}</div>'
        f'<div class="stat-label">{label}</div>'
        f"</div>"
    )


def render_ready_badge(pdf_name: str) -> str:
    """Return the green 'ready' badge HTML showing the active PDF name."""
    return f'<span class="ready-badge">✅ {pdf_name}</span>'


def render_empty_chat_state() -> str:
    """Return the HTML for the empty chat placeholder."""
    return (
        '<div class="empty-state">'
        '<div class="empty-state-icon">💬</div>'
        '<div class="empty-state-text">Ask anything about your document below</div>'
        "</div>"
    )


FOOTER_HTML: str = """
<div class="powered-by">
    🤗 <span>HuggingFace</span> &nbsp;·&nbsp;
    📦 <span>Pinecone</span> &nbsp;·&nbsp;
    ⚡ <span>Groq Llama 3</span> &nbsp;·&nbsp;
    🦜 <span>LangChain</span>
</div>
"""
