"""
DocQuery AI — Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Groq LLM Settings
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Available models — display name → Groq model ID
AVAILABLE_MODELS = {
    "Llama 3.3 70B (Best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast)": "llama-3.1-8b-instant",
    "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
    "Gemma 2 9B (Google)": "gemma2-9b-it",
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# ──────────────────────────────────────────────
# Embedding Settings
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"           # 384-dim, fast, local
EMBEDDING_DEVICE = "cpu"                        # Use "cuda" if GPU available

# ──────────────────────────────────────────────
# Chunking Settings
# ──────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ──────────────────────────────────────────────
# Retrieval Settings
# ──────────────────────────────────────────────
TOP_K = 5

# ──────────────────────────────────────────────
# File Upload Settings
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = ["pdf", "txt", "md", "docx"]
MAX_FILE_SIZE_MB = 50
MAX_FILES = 20

# ──────────────────────────────────────────────
# UI Theme
# ──────────────────────────────────────────────
ACCENT_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
ACCENT_COLOR = "#7c8aff"
BG_PRIMARY = "#0a0e17"
BG_SECONDARY = "#111827"
BG_CARD = "rgba(17, 24, 39, 0.7)"
BORDER_COLOR = "rgba(124, 138, 255, 0.15)"

# ──────────────────────────────────────────────
# System Prompt for RAG
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are DocQuery AI, a precise document analysis assistant.

RULES:
1. Answer ONLY using the provided context. Do not use prior knowledge.
2. If the context doesn't contain the answer, say: "I couldn't find this information in the uploaded documents."
3. Be concise but thorough. Use bullet points for lists.
4. When quoting from documents, mention which part of the document it comes from.
5. If asked to summarize, cover all major points from the context.
6. Use markdown formatting for better readability.

CONTEXT FROM DOCUMENTS:
{context}
"""
