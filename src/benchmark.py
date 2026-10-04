from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return payload


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    folded = answer.casefold()
    hits = sum(item.casefold() in folded for item in expected)
    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Return a lightweight correctness-and-concision score for offline mode."""

    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concision = 1.0 if len(answer) <= 500 else 0.5
    return round(0.8 * recall + 0.2 * concision, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    user_ids = {str(item["user_id"]) for item in conversations}
    size_for = getattr(agent, "memory_file_size", lambda _user: 0)
    before = sum(size_for(user) for user in user_ids)
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = f"{agent_name.lower()}-{conversation['id']}"
        thread_ids.add(thread_id)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, str(turn))

        for index, item in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, str(item["question"]))
            answer = str(result.get("response", result.get("answer", "")))
            expected = [str(value) for value in item.get("expected_contains", [])]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    after = sum(size_for(user) for user in user_ids)

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(thread_id) for thread_id in thread_ids),
        prompt_tokens_processed=sum(agent.prompt_token_usage(thread_id) for thread_id in thread_ids),
        recall_score=average(recall_scores),
        response_quality=average(quality_scores),
        memory_growth_bytes=max(0, after - before),
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render benchmark rows as a Markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(values, headers=headers, tablefmt="github")
    except ImportError:
        rendered = ["| " + " | ".join(headers) + " |"]
        rendered.append("| " + " | ".join("---" for _ in headers) + " |")
        rendered.extend("| " + " | ".join(map(str, row)) + " |" for row in values)
        return "\n".join(rendered)


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    suites = [
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    ]
    for title, path in suites:
        conversations = load_conversations(path)
        agents = [
            ("Baseline", BaselineAgent(config, force_offline=True)),
            ("Advanced", AdvancedAgent(config, force_offline=True)),
        ]
        rows = [run_agent_benchmark(name, agent, conversations, config) for name, agent in agents]
        print(f"\n## {title}\n")
        print(format_rows(rows))


if __name__ == "__main__":
    main()
