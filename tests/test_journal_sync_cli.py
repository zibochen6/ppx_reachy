"""Tests for the unattended `sync-journals` CLI command and its flock."""

from __future__ import annotations

import fcntl
import sys
from types import SimpleNamespace

import pytest

from chaihuo_reachy import main as main_module
from chaihuo_reachy.config import Config
from chaihuo_reachy.memory.journal_fetcher import journal_sync_lock


def test_build_parser_accepts_sync_journals() -> None:
    args = main_module.build_parser().parse_args(["sync-journals"])
    assert args.command == "sync-journals"


class _StubFetcher:
    """Fake JournalFetcher: records sync() calls, returns canned results."""

    def __init__(self, results=None, health=None, error=None) -> None:
        self.results = results or []
        self.health_data = health or {
            "expected": 3,
            "complete": 3,
            "last_checked_at": "",
            "last_success_at": "",
            "failures": [],
        }
        self.error = error
        self.sync_calls = 0
        self.last_kwargs: dict = {}

    async def sync(self, **kwargs):
        self.sync_calls += 1
        self.last_kwargs = kwargs
        if self.error:
            raise RuntimeError(self.error)
        return self.results

    def health(self):
        return dict(self.health_data)


class _StubStore:
    def __init__(self, **_kwargs) -> None:
        pass


def _stub_memory_classes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        main_module.sys.modules["chaihuo_reachy.memory"], "JournalFetcher", _StubFetcher
    )
    monkeypatch.setattr(
        main_module.sys.modules["chaihuo_reachy.memory"], "MemoryStore", _StubStore
    )


@pytest.mark.asyncio
async def test_sync_incremental_returns_zero_on_success(
    monkeypatch, tmp_path, capsys
) -> None:
    _stub_memory_classes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "chaihuo_reachy.memory.JournalFetcher",
        lambda **kwargs: _StubFetcher(
            results=[
                {"slug": "a", "new": True, "changed": False},
                {"slug": "b", "new": False, "changed": True},
                {"slug": "c", "new": False, "changed": False},
            ]
        ),
    )
    cfg = Config(
        journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path)
    )
    rc = await main_module._sync_journals_incremental(cfg)
    assert rc == 0
    assert "新增 1" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_incremental_returns_one_when_listing_unavailable(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "chaihuo_reachy.memory.JournalFetcher",
        lambda **kwargs: _StubFetcher(error="官方日记目录不可用，无法校验完整性"),
    )
    cfg = Config(journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path))
    rc = await main_module._sync_journals_incremental(cfg)
    assert rc == 1


@pytest.mark.asyncio
async def test_sync_incremental_partial_corpus_returns_zero(
    monkeypatch, tmp_path, capsys
) -> None:
    # The known Yuque 401-private-entry case: corpus incomplete is NOT fatal.
    monkeypatch.setattr(
        "chaihuo_reachy.memory.JournalFetcher",
        lambda **kwargs: _StubFetcher(
            results=[{"slug": "a", "new": False, "changed": False}],
            health={
                "expected": 5,
                "complete": 4,
                "last_checked_at": "",
                "last_success_at": "",
                "failures": ["private: 401"],
            },
        ),
    )
    cfg = Config(journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path))
    rc = await main_module._sync_journals_incremental(cfg)
    assert rc == 0
    out = capsys.readouterr().out
    assert "⚠️" in out
    assert "官方 5 篇" in out


@pytest.mark.asyncio
async def test_sync_incremental_skips_when_lock_held(
    monkeypatch, tmp_path, capsys
) -> None:
    lock_path = tmp_path / ".sync.lock"
    lock_fd = lock_path.open("a+")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # another sync owns the lock
        monkeypatch.setattr(
            "chaihuo_reachy.memory.JournalFetcher",
            lambda **kwargs: _StubFetcher(
                results=[{"slug": "x", "new": True, "changed": False}]
            ),
        )
        cfg = Config(
            journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path)
        )
        rc = await main_module._sync_journals_incremental(cfg)
        assert rc == 0  # periodic best-effort: skip, next tick retries
        assert "skip" in capsys.readouterr().out
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def test_journal_sync_lock_mutual_exclusion(tmp_path) -> None:
    with journal_sync_lock(tmp_path) as outer:
        assert outer is True
        with journal_sync_lock(tmp_path) as nested:
            assert nested is False  # same-process separate fd is also excluded
    with journal_sync_lock(tmp_path) as reacquired:
        assert reacquired is True


def test_main_sync_journals_exit_code_before_api_key_gate(
    monkeypatch, tmp_path, capsys
) -> None:
    """sync-journals must run without BAILIAN_API_KEY (systemd service use)."""
    monkeypatch.setattr(
        main_module, "load_config", lambda _path=None: Config(
            journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path)
        )
    )
    monkeypatch.setattr(
        "chaihuo_reachy.memory.JournalFetcher",
        lambda **kwargs: _StubFetcher(
            results=[{"slug": "a", "new": False, "changed": False}]
        ),
    )
    monkeypatch.setattr(sys, "argv", ["chaihuo-reachy", "sync-journals"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 0
    assert "日记同步完成" in capsys.readouterr().out


def test_fetcher_dispatch_exposes_sync_kwargs(monkeypatch, tmp_path) -> None:
    """The stubbed fetcher sees incremental (no refresh_all) + memory_store."""
    captured: dict = {}

    class Fetcher(_StubFetcher):
        async def sync(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr("chaihuo_reachy.memory.JournalFetcher", lambda **kw: Fetcher())
    cfg = Config(journal_cache_dir=str(tmp_path), chroma_persist_dir=str(tmp_path))
    rc = main_module.asyncio.run(main_module._sync_journals_incremental(cfg))
    assert rc == 0
    assert captured.get("refresh_all") is None  # incremental, not full
    assert "memory_store" in captured
