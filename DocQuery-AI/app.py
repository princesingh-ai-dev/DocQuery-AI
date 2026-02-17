"""
DocQuery AI — Streamlit Chat Interface
Upload PDFs/TXT/MD/DOCX, ask questions, get AI-powered streaming answers with source citations.
"""

import streamlit as st
from rag_engine import VectorStore, extract_text, chunk_documents, rag_query_stream
from utils import format_chat_as_markdown, format_file_size, get_file_extension, truncate_text
import config


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="DocQuery AI — Chat with your Documents",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Base Theme ─── */
    .stApp {
        background: #0a0e17;
        font-family: 'Inter', sans-serif;
    }

    /* ─── Gradient Header ─── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        animation: fadeInDown 0.8s ease;
    }

    .sub-header {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: -8px;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.8s ease;
    }

    /* ─── Animations ─── */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ─── Glassmorphism Cards ─── */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(124, 138, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.6s ease;
    }
    .glass-card:hover {
        border-color: rgba(124, 138, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }

    /* ─── Feature Cards ─── */
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
        display: block;
    }
    .feature-title {
        color: #e2e8f0;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .feature-desc {
        color: rgba(255,255,255,0.45);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ─── Source Citation Cards ─── */
    .source-card {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(124, 138, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.85rem;
        line-height: 1.5;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(124, 138, 255, 0.25);
        background: rgba(17, 24, 39, 0.7);
    }
    .source-header {
        color: #7c8aff;
        font-weight: 600;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }
    .source-text {
        color: rgba(255,255,255,0.6);
        font-size: 0.82rem;
    }

    /* ─── Stats Badges ─── */
    .stat-badge {
        display: inline-block;
        background: rgba(124, 138, 255, 0.1);
        border: 1px solid rgba(124, 138, 255, 0.15);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #7c8aff;
        margin: 3px 4px;
        transition: all 0.2s ease;
    }
    .stat-badge:hover {
        background: rgba(124, 138, 255, 0.18);
        transform: scale(1.05);
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid rgba(124, 138, 255, 0.08);
    }

    .sidebar-logo {
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sidebar-tagline {
        color: rgba(255,255,255,0.4);
        font-size: 0.82rem;
        margin-bottom: 12px;
    }

    /* ─── Analytics Card ─── */
    .analytics-card {
        background: rgba(124, 138, 255, 0.06);
        border: 1px solid rgba(124, 138, 255, 0.12);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }
    .analytics-title {
        color: #7c8aff;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    .analytics-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .analytics-row:last-child {
        border-bottom: none;
    }
    .analytics-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
    }
    .analytics-value {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* ─── Chat Avatars ─── */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #00d4aa, #00b4d8) !important;
    }

    /* ─── Divider ─── */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124, 138, 255, 0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ─── Onboarding Suggestion ─── */
    .suggestion-chip {
        display: inline-block;
        background: rgba(124, 138, 255, 0.08);
        border: 1px solid rgba(124, 138, 255, 0.15);
        border-radius: 24px;
        padding: 8px 18px;
        margin: 4px;
        color: rgba(255,255,255,0.7);
        font-size: 0.85rem;
        cursor: default;
        transition: all 0.2s ease;
    }
    .suggestion-chip:hover {
        background: rgba(124, 138, 255, 0.15);
        color: #fff;
        transform: translateY(-1px);
    }

    /* ─── Processing Indicator ─── */
    .processing-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #7c8aff;
        animation: pulse 1.5s infinite;
        margin: 0 2px;
    }

    /* ─── Hide streamlit chrome ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔍 DocQuery AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Intelligent document analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # API Key
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=config.GROQ_API_KEY,
        help="Get your free key at [console.groq.com](https://console.groq.com/keys)",
        placeholder="gsk_...",
    )

    # Model selection
    model_display = st.selectbox(
        "🧠 AI Model",
        options=list(config.AVAILABLE_MODELS.keys()),
        index=0,
        help="Choose the LLM for generating answers",
    )
    selected_model = config.AVAILABLE_MODELS[model_display]

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # File Upload
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=config.SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help=f"Supports: {', '.join(ext.upper() for ext in config.SUPPORTED_EXTENSIONS)}",
    )

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                file_bytes = uploaded_file.read()

                # Duplicate check
                if st.session_state.vector_store.is_duplicate(file_bytes):
                    st.info(f"📎 `{uploaded_file.name}` already loaded")
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    continue

                with st.spinner(f"Processing `{uploaded_file.name}`..."):
                    # Extract text (multi-format)
                    pages = extract_text(file_bytes, uploaded_file.name)

                    if not pages:
                        st.warning(f"⚠️ No text found in `{uploaded_file.name}`")
                        continue

                    # Chunk
                    chunks = chunk_documents(pages)

                    # Embed + index
                    st.session_state.vector_store.add_documents(
                        chunks,
                        file_content=file_bytes,
                        filename=uploaded_file.name,
                        num_pages=len(pages),
                    )
                    st.session_state.uploaded_files.append(uploaded_file.name)

                    ext = get_file_extension(uploaded_file.name).upper()
                    st.success(
                        f"✅ `{uploaded_file.name}` — {len(pages)} {'pages' if ext == 'PDF' else 'sections'}, {len(chunks)} chunks"
                    )

    # Analytics Dashboard
    vs = st.session_state.vector_store
    if vs.total_chunks > 0:
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        stats = vs.get_stats()

        analytics_html = f"""
        <div class="analytics-card">
            <div class="analytics-title">📊 Knowledge Base</div>
            <div class="analytics-row">
                <span class="analytics-label">Documents</span>
                <span class="analytics-value">{stats['documents']}</span>
            </div>
            <div class="analytics-row">
                <span class="analytics-label">Total Chunks</span>
                <span class="analytics-value">{stats['chunks']:,}</span>
            </div>
            <div class="analytics-row">
                <span class="analytics-label">Pages / Sections</span>
                <span class="analytics-value">{stats['total_pages']}</span>
            </div>
            <div class="analytics-row">
                <span class="analytics-label">≈ Tokens</span>
                <span class="analytics-value">{stats['estimated_tokens']:,}</span>
            </div>
        </div>
        """
        st.markdown(analytics_html, unsafe_allow_html=True)

        # File type badges
        if stats["file_types"]:
            badges = "".join(
                f'<span class="stat-badge">{ft} × {count}</span>'
                for ft, count in stats["file_types"].items()
            )
            st.markdown(badges, unsafe_allow_html=True)

        # Loaded files list
        if stats["filenames"]:
            with st.expander("📁 Loaded Files", expanded=False):
                for fn in stats["filenames"]:
                    st.markdown(f"📄 `{fn}`")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Settings
    with st.expander("⚙️ Settings", expanded=False):
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0,
            value=config.LLM_TEMPERATURE, step=0.1,
            help="Lower = more factual, Higher = more creative",
        )

        top_k = st.slider(
            "Context chunks (Top-K)",
            min_value=1, max_value=10,
            value=config.TOP_K,
            help="Number of document chunks to retrieve per question",
        )

    # Actions strip
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    if col1.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if col2.button("🔄 Reset All", use_container_width=True):
        st.session_state.vector_store = VectorStore()
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.rerun()

    # Chat export
    if st.session_state.messages:
        md_export = format_chat_as_markdown(st.session_state.messages)
        st.download_button(
            label="📥 Download Chat",
            data=md_export,
            file_name="docquery_chat.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────

# Header
st.markdown('<p class="main-header">🔍 DocQuery AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload documents → Ask questions → Get AI-powered answers with citations</p>',
    unsafe_allow_html=True,
)

# Welcome screen (no docs uploaded)
if vs.total_chunks == 0:
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="glass-card">'
            '<span class="feature-icon">📄</span>'
            '<div class="feature-title">Upload Documents</div>'
            '<div class="feature-desc">Drop PDF, TXT, Markdown, or DOCX files in the sidebar to build your knowledge base</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="glass-card">'
            '<span class="feature-icon">💬</span>'
            '<div class="feature-title">Ask Questions</div>'
            '<div class="feature-desc">Type any question about your documents in natural language — follow-ups work too</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="glass-card">'
            '<span class="feature-icon">🎯</span>'
            '<div class="feature-title">Get Cited Answers</div>'
            '<div class="feature-desc">AI analyzes your docs and streams answers with exact page &amp; source citations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Onboarding suggestion chips
    st.markdown("**💡 Example questions you can ask:**")
    chips = [
        "Summarize the main points",
        "What does it say about ...?",
        "Compare sections 3 and 5",
        "List all key metrics",
        "Explain this concept in simple terms",
    ]
    chips_html = "".join(f'<span class="suggestion-chip">{c}</span>' for c in chips)
    st.markdown(f'<div style="text-align:center;margin:12px 0;">{chips_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.info("👈 **Start by uploading a document in the sidebar** and entering your Groq API key")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"📚 Sources ({len(message['sources'])} chunks)", expanded=False):
                for src in message["sources"]:
                    score_pct = f"{src['score']:.0%}" if src.get("score") else ""
                    preview = truncate_text(src["text"], 300)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">📄 {src["source"]} — Page {src["page"]} '
                        f'<span style="color:#4CAF50">({score_pct} match)</span></div>'
                        f'<div class="source-text">{preview}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=(vs.total_chunks == 0),
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate streaming response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents..."):
            stream, sources = rag_query_stream(
                question=prompt,
                vector_store=st.session_state.vector_store,
                chat_history=st.session_state.chat_history,
                top_k=top_k if "top_k" in dir() else config.TOP_K,
                temperature=temperature if "temperature" in dir() else config.LLM_TEMPERATURE,
                api_key=api_key if api_key else None,
                model=selected_model,
            )

        # Stream the response
        answer = st.write_stream(stream)

        # Show source cards
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} chunks)", expanded=False):
                for src in sources:
                    score_pct = f"{src['score']:.0%}" if src.get("score") else ""
                    preview = truncate_text(src["text"], 300)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">📄 {src["source"]} — Page {src["page"]} '
                        f'<span style="color:#4CAF50">({score_pct} match)</span></div>'
                        f'<div class="source-text">{preview}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
