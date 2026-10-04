from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B with persistent profile and compact thread memory.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between deterministic offline mode and the live model."""

        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)

        self._remember_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.compact_memory.context(thread_id)
        system = (
            "Use this persistent user profile and compact conversation summary when answering.\n"
            f"PROFILE:\n{self.profile_store.read_text(user_id)}\n"
            f"SUMMARY:\n{context['summary']}"
        )
        messages = [{"role": "system", "content": system}, *context["messages"]]
        result = self.langchain_agent.invoke(messages)
        response = _message_text(result)
        self.compact_memory.append(thread_id, "assistant", response)
        generated = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return _result(response, generated, prompt_tokens)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        self._remember_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        generated = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return _result(response, generated, prompt_tokens)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        pieces = [self.profile_store.read_text(user_id), str(context["summary"])]
        pieces.extend(str(item.get("content", "")) for item in context["messages"])
        return estimate_tokens("\n".join(pieces))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin bền vững nào về bạn."

        lower = message.lower()
        requested: list[tuple[str, str]] = []
        field_cues = {
            "name": ("tên", "ai"),
            "profession": ("nghề", "làm gì"),
            "location": ("ở đâu", "nơi ở", "huế", "hà nội"),
            "response_style": ("style", "kiểu trả lời"),
            "favorite_drink": ("đồ uống", "uống"),
            "favorite_food": ("món ăn",),
            "pet": ("nuôi", "con gì"),
            "interests": ("mối quan tâm", "quan tâm", "tóm tắt"),
        }
        labels = {
            "name": "tên",
            "profession": "nghề nghiệp hiện tại",
            "location": "nơi ở hiện tại",
            "response_style": "style trả lời",
            "favorite_drink": "đồ uống yêu thích",
            "favorite_food": "món ăn yêu thích",
            "pet": "thú cưng",
            "interests": "mối quan tâm",
        }
        for key, cues in field_cues.items():
            if key in facts and any(cue in lower for cue in cues):
                requested.append((labels[key], facts[key]))
        if not requested:
            return "Mình đã cập nhật memory bền vững cho bạn."
        return "; ".join(f"{label}: {value}" for label, value in requested) + "."

    def _maybe_build_langchain_agent(self):
        """Build the configured live chat model when credentials are available.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        provider = self.config.model.provider
        has_credentials = bool(self.config.model.api_key or provider == "ollama")
        if not has_credentials:
            return None
        try:
            return build_chat_model(self.config.model)
        except (ImportError, ValueError):
            return None

    def _remember_profile_updates(self, user_id: str, message: str) -> None:
        current = self.profile_store.facts(user_id)
        for key, value in extract_profile_updates(message).items():
            if key in {"interests", "response_style"} and key in current:
                old_parts = [part.strip() for part in current[key].split(",")]
                new_parts = [part.strip() for part in value.split(",")]
                value = ", ".join(dict.fromkeys(part for part in old_parts + new_parts if part))
            self.profile_store.upsert_fact(user_id, key, value)
            current[key] = value


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
