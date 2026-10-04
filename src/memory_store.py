from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a stable character-count heuristic.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    normalized = text.strip()
    if not normalized:
        return 0
    # Four characters per token is deliberately simple, stable, and language agnostic.
    return max(1, (len(normalized) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    The store:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        raw = unicodedata.normalize("NFKC", user_id).strip()
        slug = re.sub(r"[^\w.-]+", "-", raw, flags=re.UNICODE).strip(".-_")
        if not slug or slug in {".", ".."}:
            raise ValueError("user_id must contain at least one safe character")
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User Profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.rstrip() + "\n"
        path.write_text(normalized, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        original = self.read_text(user_id)
        if search_text not in original:
            return False
        self.write_text(user_id, original.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the simple ``- key: value`` profile format."""

        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s*([\w_-]+)\s*:\s*(.+?)\s*$", line)
            if match:
                facts[match.group(1)] = match.group(2)
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        facts = self.facts(user_id)
        facts[key] = value.strip().strip(" .")
        lines = ["# User Profile", "", *[f"- {name}: {fact}" for name, fact in facts.items()]]
        return self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into high-confidence stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join(message.strip().split())
    if not text:
        return {}
    # A question can mention a profile field without asserting its value
    # (for example, "Mình tên gì?"). Never learn persistent facts from it.
    if "?" in text:
        return {}
    if re.match(
        r"^(?:nhắc|tóm tắt|bạn (?:biết|có biết|thử nhớ)|nếu (?:ai|phải)|hiện tại mình đang ở)",
        text,
        re.IGNORECASE,
    ):
        return {}

    updates: dict[str, str] = {}

    def capture(key: str, patterns: list[str]) -> None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ,.;:!?\"'")
                if value.lower().startswith("và ") or re.search(
                    r"\b(?:gì|đâu|nào|ai)\b", value, re.IGNORECASE
                ):
                    continue
                if value:
                    updates[key] = value
                return

    capture(
        "name",
        [r"\b(?:mình|tôi)\s+tên\s+(?:là\s+)?([^,.!?]+?)(?=\s+và\s+|[,.!?]|$)"],
    )
    capture(
        "location",
        [
            r"\bnơi ở (?:hiện tại|đã cập nhật)[^,.!?]*?(?:là|sang)\s+([^,.!?]+)",
            r"\b(?:hiện|đang)\s+ở\s+([^,.!?]+?)(?=\s+(?:để|chứ|dù|vì)\s+|[,.!?]|$)",
            r"\bgiờ\s+(?:mình|tôi)\s+đang\s+ở\s+([^,.!?]+)",
            r"\b(?:mình|tôi)\s+vẫn\s+ở\s+([^,.!?]+?)(?=\s+(?:để|chứ|dù|vì)\s+|[,.!?]|$)",
            r"\b(?:mình|tôi)\s+ở\s+([^,.!?]+?)(?=\s+và\s+|[,.!?]|$)",
        ],
    )
    capture(
        "profession",
        [
            r"\bnghề nghiệp (?:hiện tại )?(?:thì )?vẫn là\s+([^,.!?]+)",
            r"\bnghề nghiệp hiện tại[^,.!?]*?là\s+([^,.!?]+)",
            r"\bgiờ (?:(?:mình|tôi) )?(?:đã )?chuyển sang\s+([^,.!?]+)",
            r"\b(?:mình|tôi)[^,.!?]*?\b(?:hiện\s+)?đang làm\s+([^,.!?]+?)(?=\s+cho\s+|[,.!?]|$)",
            r"\b(?:mình|tôi) (?:hiện )?đang làm\s+([^,.!?]+?)(?=\s+cho\s+|[,.!?]|$)",
        ],
    )
    capture("favorite_drink", [r"\b(?:đồ uống yêu thích|món uống yêu thích)\s+(?:của mình\s+)?là\s+([^,.!?]+)"])
    capture("favorite_food", [r"\bmón ăn yêu thích\s+(?:của mình\s+)?là\s+([^,.!?]+)"])
    capture("pet", [r"\b(?:mình|tôi) nuôi\s+(?:một\s+)?([^,.!?]+)"])

    if re.search(r"\b(?:mình|tôi) (?:thích|đang quan tâm (?:nhiều )?đến)\s+", text, re.IGNORECASE):
        match = re.search(
            r"\b(?:mình|tôi) (?:thích|đang quan tâm (?:nhiều )?đến)\s+([^.!?]+)",
            text,
            re.IGNORECASE,
        )
        if match and not re.search(r"\b(?:trả lời|cách giải thích|kiểu trả lời)\b", match.group(1), re.IGNORECASE):
            updates["interests"] = match.group(1).strip(" ,.;")

    style_cues: list[str] = []
    lower = text.lower()
    if "3 bullet" in lower:
        style_cues.append("3 bullet ngắn")
    elif "bullet" in lower:
        style_cues.append("bullet ngắn")
    if "ngắn gọn" in lower or "trả lời ngắn" in lower or "câu trả lời ngắn" in lower:
        if not style_cues:
            style_cues.append("ngắn gọn")
    if "ví dụ thực chiến" in lower:
        style_cues.append("có ví dụ thực chiến")
    elif "ví dụ thực tế" in lower:
        style_cues.append("có ví dụ thực tế")
    if "trade-off" in lower and re.search(r"(?:thích|muốn|ưu tiên|style)", lower):
        style_cues.append("nhấn mạnh trade-off")
    if style_cues and re.search(r"(?:mình|tôi).*(?:muốn|thích|ưu tiên)|style trả lời", lower):
        updates["response_style"] = ", ".join(dict.fromkeys(style_cues))

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded heuristic summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if max_items <= 0 or not messages:
        return ""
    snippets: list[str] = []
    for item in messages[-max_items:]:
        role = item.get("role", "unknown")
        content = " ".join(item.get("content", "").split())
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        if content:
            snippets.append(f"{role}: {content}")
    return "\n".join(snippets)


@dataclass
class CompactMemoryManager:
    """Keep recent messages and compact older messages into a summary.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread["summary"])
        total_text = summary + "\n" + "\n".join(
            str(item.get("content", "")) for item in messages
        )
        if estimate_tokens(total_text) <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages
        older = messages[:split_at]
        summary_input: list[dict[str, str]] = []
        if summary:
            summary_input.append({"role": "summary", "content": summary})
        summary_input.extend(older)
        thread["summary"] = summarize_messages(summary_input, max_items=max(2, self.keep_messages))
        thread["messages"] = messages[split_at:]
        thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        return {
            "messages": [dict(item) for item in thread["messages"]],
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
