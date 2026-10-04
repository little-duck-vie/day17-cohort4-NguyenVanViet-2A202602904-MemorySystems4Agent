from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a complete ``LabConfig``.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM", default_provider="openai", default_model="gpt-4o-mini")
    judge_model = _provider_config(
        "JUDGE",
        default_provider=model.provider,
        default_model=model.model_name,
    )
    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=_env_positive_int("COMPACT_THRESHOLD_TOKENS", 1_200),
        compact_keep_messages=_env_positive_int("COMPACT_KEEP_MESSAGES", 6),
        model=model,
        judge_model=judge_model,
    )


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _provider_config(prefix: str, default_provider: str, default_model: str) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    model_name = os.getenv(f"{prefix}_MODEL", default_model)
    try:
        temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))
    except ValueError as exc:
        raise ValueError(f"{prefix}_TEMPERATURE must be a number.") from exc

    provider_prefix = {
        "openai": "OPENAI",
        "custom": "CUSTOM",
        "gemini": "GEMINI",
        "anthropic": "ANTHROPIC",
        "ollama": "OLLAMA",
        "openrouter": "OPENROUTER",
    }[provider]
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{provider_prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{provider_prefix}_BASE_URL")
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
