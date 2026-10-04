from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(tmp_path)
    config.compact_threshold_tokens = 80
    config.compact_keep_messages = 4
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    path = store.write_text("user/unsafe", "# User Profile\n\n- name: Dũng")
    assert path.exists()
    assert "Dũng" in store.read_text("user/unsafe")
    assert store.edit_text("user/unsafe", "Dũng", "DungCT") is True
    assert "DungCT" in store.read_text("user/unsafe")
    assert store.file_size("user/unsafe") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply("user", "thread", f"Lượt {index}: " + "nội dung dài " * 20)
    context = agent.compact_memory.context("thread")
    assert agent.compaction_count("thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    statement = "Mình tên là DũngCT và hiện đang làm MLOps engineer."
    advanced.reply("dung", "old", statement)
    baseline.reply("dung", "old", statement)

    advanced_answer = advanced.reply("dung", "new", "Mình tên gì và làm nghề gì?")["response"]
    baseline_answer = baseline.reply("dung", "new", "Mình tên gì và làm nghề gì?")["response"]
    assert "DũngCT" in advanced_answer and "MLOps engineer" in advanced_answer
    assert "DũngCT" not in baseline_answer and "MLOps engineer" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(25):
        message = f"Lượt {index}: " + "đây là ngữ cảnh dài dùng để kiểm thử compact memory. " * 12
        advanced.reply("user", "long", message)
        baseline.reply("user", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
