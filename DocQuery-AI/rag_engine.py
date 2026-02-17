"""
DocQuery AI — RAG Engine
PDF/TXT/MD/DOCX loading → chunking → embedding → retrieval → LLM generation (streaming)
"""

import fitz                                     # PyMuPDF
import numpy as np
import faiss
import hashlib
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils import get_file_extension, estimate_tokens


# ──────────────────────────────────────────────
# Text Extraction — Multi-Format
# ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document") -> list[dict]:
    """Extract text from PDF bytes, returning list of {page, text, source} dicts."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "source": filename,
            })
    doc.close()
    return pages


def extract_text_from_txt(file_bytes: bytes, filename: str = "document") -> list[dict]:
    """Extract text from plain text file."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    # Treat the whole file as one "page"
    return [{"page": 1, "text": text, "source": filename}]


def extract_text_from_markdown(file_bytes: bytes, filename: str = "document") -> list[dict]:
    """Extract text from Markdown file."""
    text = file_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    # Split by H1/H2 headings as logical sections
    sections = []
    current_section = []
    page_num = 1

    for line in text.split("\n"):
        if line.startswith("# ") and current_section:
            sections.append({
                "page": page_num,
                "text": "\n".join(current_section).strip(),
                "source": filename,
            })
            page_num += 1
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append({
            "page": page_num,
            "text": "\n".join(current_section).strip(),
            "source": filename,
        })

    return sections if sections else [{"page": 1, "text": text, "source": filename}]


def extract_text_from_docx(file_bytes: bytes, filename: str = "document") -> list[dict]:
    """Extract text from DOCX file."""
    try:
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return []
        full_text = "\n\n".join(paragraphs)
        return [{"page": 1, "text": full_text, "source": filename}]
    except Exception as e:
        return []


def extract_text(file_bytes: bytes, filename: str) -> list[dict]:
    """Unified text extraction dispatcher by file extension."""
    ext = get_file_extension(filename)

    extractors = {
        "pdf": extract_text_from_pdf,
        "txt": extract_text_from_txt,
        "md": extract_text_from_markdown,
        "docx": extract_text_from_docx,
    }

    extractor = extractors.get(ext)
    if extractor is None:
        return []

    return extractor(file_bytes, filename)


# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────

def chunk_documents(
    pages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """Split page texts into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for page_info in pages:
        splits = splitter.split_text(page_info["text"])
        for j, chunk_text in enumerate(splits):
            chunks.append({
                "text": chunk_text,
                "source": page_info["source"],
                "page": page_info["page"],
                "chunk_id": f"{page_info['source']}_p{page_info['page']}_c{j}",
            })

    return chunks


# ──────────────────────────────────────────────
# Embedding + Vector Store
# ──────────────────────────────────────────────

class VectorStore:
    """FAISS-backed vector store with sentence-transformer embeddings."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, device=config.EMBEDDING_DEVICE)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self._doc_hashes = set()
        self._doc_metadata = {}  # hash → {filename, pages, file_type, size}

    def _file_hash(self, content: bytes) -> str:
        """Generate hash to detect duplicate uploads."""
        return hashlib.md5(content).hexdigest()

    def is_duplicate(self, content: bytes) -> bool:
        """Check if this file has already been indexed."""
        return self._file_hash(content) in self._doc_hashes

    def add_documents(
        self,
        chunks: list[dict],
        file_content: bytes = None,
        filename: str = "",
        num_pages: int = 0,
    ):
        """Embed chunks and add to FAISS index."""
        if file_content:
            h = self._file_hash(file_content)
            self._doc_hashes.add(h)
            self._doc_metadata[h] = {
                "filename": filename,
                "pages": num_pages,
                "file_type": get_file_extension(filename),
                "size": len(file_content),
            }

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """Find the most relevant chunks for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self._doc_hashes)

    def get_stats(self) -> dict:
        """Return detailed knowledge base statistics."""
        total_chars = sum(len(c["text"]) for c in self.chunks)
        file_types = {}
        total_pages = 0
        filenames = []

        for meta in self._doc_metadata.values():
            ft = meta["file_type"].upper()
            file_types[ft] = file_types.get(ft, 0) + 1
            total_pages += meta["pages"]
            filenames.append(meta["filename"])

        return {
            "documents": self.total_documents,
            "chunks": self.total_chunks,
            "total_chars": total_chars,
            "estimated_tokens": estimate_tokens("x" * total_chars),
            "total_pages": total_pages,
            "file_types": file_types,
            "filenames": filenames,
        }

    def clear(self):
        """Reset the vector store."""
        self.index = None
        self.chunks = []
        self._doc_hashes = set()
        self._doc_metadata = {}


# ──────────────────────────────────────────────
# LLM Query — Standard (blocking)
# ──────────────────────────────────────────────

def query_llm(
    question: str,
    context_chunks: list[dict],
    chat_history: list[dict] = None,
    temperature: float = config.LLM_TEMPERATURE,
    api_key: str = None,
    model: str = None,
) -> str:
    """Send question + context to Groq LLM (blocking)."""

    key = api_key or config.GROQ_API_KEY
    if not key:
        return "❌ **Groq API key not set.** Add your key in the sidebar or create a `.env` file."

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_info = f"[Source: {chunk['source']}, Page {chunk['page']}]"
        context_parts.append(f"--- Chunk {i} {source_info} ---\n{chunk['text']}")

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    system_message = config.SYSTEM_PROMPT.format(context=context_text)

    messages = [{"role": "system", "content": system_message}]
    if chat_history:
        for msg in chat_history[-12:]:
            messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=model or config.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=config.LLM_MAX_TOKENS,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return _format_error(e)


# ──────────────────────────────────────────────
# LLM Query — Streaming
# ──────────────────────────────────────────────

def query_llm_stream(
    question: str,
    context_chunks: list[dict],
    chat_history: list[dict] = None,
    temperature: float = config.LLM_TEMPERATURE,
    api_key: str = None,
    model: str = None,
):
    """Stream answer from Groq LLM. Yields string chunks."""

    key = api_key or config.GROQ_API_KEY
    if not key:
        yield "❌ **Groq API key not set.** Add your key in the sidebar or create a `.env` file."
        return

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_info = f"[Source: {chunk['source']}, Page {chunk['page']}]"
        context_parts.append(f"--- Chunk {i} {source_info} ---\n{chunk['text']}")

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    system_message = config.SYSTEM_PROMPT.format(context=context_text)

    messages = [{"role": "system", "content": system_message}]
    if chat_history:
        for msg in chat_history[-12:]:
            messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        client = Groq(api_key=key)
        stream = client.chat.completions.create(
            model=model or config.DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=config.LLM_MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        yield _format_error(e)


# ──────────────────────────────────────────────
# Error Formatting
# ──────────────────────────────────────────────

def _format_error(e: Exception) -> str:
    error_msg = str(e)
    if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
        return "❌ **Invalid API key.** Please check your Groq API key."
    elif "rate" in error_msg.lower():
        return "⏳ **Rate limited.** Please wait a moment and try again."
    elif "model" in error_msg.lower() and "not found" in error_msg.lower():
        return "❌ **Model not available.** Try selecting a different model in settings."
    else:
        return f"❌ **Error:** {error_msg}"


# ──────────────────────────────────────────────
# Full RAG Pipeline
# ──────────────────────────────────────────────

def rag_query(
    question: str,
    vector_store: VectorStore,
    chat_history: list[dict] = None,
    top_k: int = config.TOP_K,
    temperature: float = config.LLM_TEMPERATURE,
    api_key: str = None,
    model: str = None,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline (blocking): retrieve + generate."""
    relevant_chunks = vector_store.search(question, top_k=top_k)
    answer = query_llm(
        question=question,
        context_chunks=relevant_chunks,
        chat_history=chat_history,
        temperature=temperature,
        api_key=api_key,
        model=model,
    )
    return answer, relevant_chunks


def rag_query_stream(
    question: str,
    vector_store: VectorStore,
    chat_history: list[dict] = None,
    top_k: int = config.TOP_K,
    temperature: float = config.LLM_TEMPERATURE,
    api_key: str = None,
    model: str = None,
):
    """Full RAG pipeline (streaming): retrieve + stream generate.
    Returns (stream_generator, source_chunks)."""
    relevant_chunks = vector_store.search(question, top_k=top_k)
    stream = query_llm_stream(
        question=question,
        context_chunks=relevant_chunks,
        chat_history=chat_history,
        temperature=temperature,
        api_key=api_key,
        model=model,
    )
    return stream, relevant_chunks
