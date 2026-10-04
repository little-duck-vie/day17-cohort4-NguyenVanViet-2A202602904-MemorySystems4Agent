from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a canonical provider name and reject unsupported values."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_ai": "openai",
        "open_router": "openrouter",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported LLM provider {value!r}. Choose one of: {choices}.")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate the chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        if config.api_key:
            common["api_key"] = config.api_key
        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOpenAI(**common)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            common["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**common)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if config.api_key:
            common["api_key"] = config.api_key
        return ChatAnthropic(**common)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOllama(**common)

    from langchain_openrouter import ChatOpenRouter

    if config.api_key:
        common["api_key"] = config.api_key
    if config.base_url:
        common["base_url"] = config.base_url
    return ChatOpenRouter(**common)
