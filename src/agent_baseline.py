from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A with within-session memory only.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        response = _message_text(result)
        session.messages.append({"role": "assistant", "content": response})
        generated = estimate_tokens(response)
        session.token_usage += generated
        session.prompt_tokens_processed += prompt_tokens
        return _result(response, generated, prompt_tokens)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Provide deterministic within-thread behavior without an API.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)

        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        response = _offline_answer(message, facts)

        session.messages.append({"role": "assistant", "content": response})
        generated = estimate_tokens(response)
        session.token_usage += generated
        session.prompt_tokens_processed += prompt_tokens
        return _result(response, generated, prompt_tokens)

    def _maybe_build_langchain_agent(self):
        """Build the configured live chat model when credentials are available.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        provider = self.config.model.provider
        has_credentials = bool(self.config.model.api_key or provider == "ollama")
        if not has_credentials:
            return None
        try:
            return build_chat_model(self.config.model)
        except (ImportError, ValueError):
            return None


def _offline_answer(message: str, facts: dict[str, str]) -> str:
    lower = message.lower()
    requested: list[str] = []
    field_cues = {
        "name": ("tên", "ai"),
        "location": ("ở đâu", "nơi ở"),
        "profession": ("nghề", "làm gì"),
        "response_style": ("style", "kiểu trả lời"),
        "favorite_drink": ("đồ uống", "uống"),
        "favorite_food": ("món ăn",),
        "pet": ("nuôi", "con gì"),
        "interests": ("mối quan tâm", "quan tâm", "thích"),
    }
    for key, cues in field_cues.items():
        if key in facts and any(cue in lower for cue in cues):
            requested.append(facts[key])
    if requested:
        return "Mình nhớ trong cuộc trò chuyện này: " + "; ".join(dict.fromkeys(requested)) + "."
    if facts:
        return "Mình đã ghi nhận thông tin này trong phiên hiện tại."
    return "Mình chưa có thông tin đó trong phiên hiện tại."


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content)


def _result(response: str, generated: int, prompt: int) -> dict[str, Any]:
    return {
        "response": response,
        "answer": response,
        "tokens": generated,
        "prompt_tokens": prompt,
    }
