"""
DocQuery AI — Utility Functions
"""

import os
from datetime import datetime


def get_file_extension(filename: str) -> str:
    """Return lowercase file extension without dot."""
    return os.path.splitext(filename)[1].lstrip(".").lower()


def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def format_chat_as_markdown(messages: list[dict]) -> str:
    """Export chat history as a Markdown string."""
    lines = [
        f"# DocQuery AI — Chat Export",
        f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "---\n",
    ]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            lines.append(f"### 🧑‍💻 You\n{content}\n")
        elif role == "assistant":
            lines.append(f"### 🤖 DocQuery AI\n{content}\n")

            # Include sources if present
            if "sources" in msg and msg["sources"]:
                lines.append("**📚 Sources:**\n")
                for src in msg["sources"]:
                    score = f"{src['score']:.0%}" if src.get("score") else "—"
                    lines.append(
                        f"- **{src['source']}** (Page {src['page']}, {score} match)"
                    )
                lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English)."""
    return len(text) // 4
