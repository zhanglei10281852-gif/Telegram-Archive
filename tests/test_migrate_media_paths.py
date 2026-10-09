"""Safety and recovery tests for ``scripts/migrate_media_paths.py``.

The old script destroyed data two ways: when a target folder already held
a same-name file it deleted the source without comparing content, and a
process killed between the folder rename and the single end-of-run database
commit left the disk renamed while every row still pointed at the old
folder - and a rerun skipped that chat forever, because the old folder
was gone.

These tests pin the replacement contract:

* one read-only plan covers chat folders, every database path column and
  avatar files, and ``--dry-run`` builds that same plan and writes
  nothing;
* a same-name target is removed only when its bytes hash equal;
  different content, type mismatches and unreadable entries stop the run
  before ANY destructive operation;
* each chat is a filesystem-then-database boundary: a crash or a
  failed stage never reports completion, and a rerun resumes from the
  actual state and converges to what one successful run would have produced;
* it works on pre-v6 archives (``messages.media_path``), on the
  current schema (media/media_versions, no such column), on already
  migrated archives, and on archives where only avatars remain.

The database assertions run against SQLite here and, via ``real_db``,
against PostgreSQL when a server is configured in conftest.
"""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "migrate_media_paths_under_test", REPO / "scripts" / "migrate_media_paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclass KW_ONLY resolution looks the defining module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script()


# ---------------------------------------------------------------------------
# Archive builders
# ---------------------------------------------------------------------------


async def _connect(db_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    return engine


async def _make_legacy_db(db_path: Path, chats: list[int], media_paths=(), message_paths=()):
    """Smallest pre-v6 schema: messages still carries media_path, no media_versions."""
    engine = await _connect(db_path)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE chats (id INTEGER PRIMARY KEY, type TEXT)"))
        await conn.execute(text("CREATE TABLE messages (id INTEGER, chat_id INTEGER, media_path TEXT)"))
        await conn.execute(text("CREATE TABLE media (id TEXT PRIMARY KEY, chat_id INTEGER, file_path TEXT)"))
        for chat_id in chats:
            await conn.execute(
                text("INSERT INTO chats (id, type) VALUES (:id, 'group')"), {"id": chat_id}
            )
        for index, path in enumerate(message_paths):
            await conn.execute(
                text("INSERT INTO messages (id, chat_id, media_path) VALUES (:id, :cid, :p)"),
                {"id": index, "cid": chats[0], "p": path},
            )
        for index, path in enumerate(media_paths):
            await conn.execute(
                text("INSERT INTO media (id, chat_id, file_path) VALUES (:id, :cid, :p)"),
                {"id": f"m{index}", "cid": chats[0], "p": path},
            )
    await engine.dispose()
    return f"sqlite+aiosqlite:///{db_path}"


async def _scalar(engine, sql, **params):
    async with engine.connect() as conn:
        return (await conn.execute(text(sql), params)).scalar()


async def _all_paths(engine):
    async with engine.connect() as conn:
        media = [r[0] for r in (await conn.execute(text("SELECT file_path FROM media"))).fetchall()]
        versions = []
        try:
            versions = [
                r[0] for r in (await conn.execute(text("SELECT file_path FROM media_versions"))).fetchall()
            ]
        except Exception:
            pass
        messages = []
        try:
            messages = [
                r[0] for r in (await conn.execute(text("SELECT media_path FROM messages"))).fetchall()
            ]
        except Exception:
            pass
    return media, versions, messages


# ---------------------------------------------------------------------------
# Plan / dry-run
# ---------------------------------------------------------------------------


async def test_dry_run_builds_the_full_plan_and_writes_nothing(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo-a")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db, [-352], media_paths=["352/a.jpg"], message_paths=["/srv/media/352/a.jpg"]
    )
    (media / "avatars" / "chats").mkdir(parents=True)
    (media / "avatars" / "chats" / "352_9.jpg").write_bytes(b"avatar")

    result = await mod.migrate(url, str(media), dry_run=True)

    assert result.status == "dry-run"
    assert (old / "a.jpg").exists()
    assert not (media / "-352").exists()
    assert (media / "avatars" / "chats" / "352_9.jpg").exists()
    engine = await _connect(db)
    try:
        media_paths, _, message_paths = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert media_paths == ["352/a.jpg"]
    assert message_paths == ["/srv/media/352/a.jpg"]


async def test_plan_counts_folders_files_db_rows_and_avatars(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"a")
    (old / "b.jpg").write_bytes(b"b")
    (media / "avatars" / "chats").mkdir(parents=True)
    (media / "avatars" / "chats" / "352_9.jpg").write_bytes(b"avatar")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db, [-352], media_paths=["352/a.jpg", "352/b.jpg"], message_paths=["/x/352/a.jpg"]
    )

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            plan = await mod.build_plan(conn, str(media))
    finally:
        await engine.dispose()

    chat = plan.chats[0]
    assert chat.rename_whole is True
    assert chat.db_rows["media.file_path"] == 2
    assert chat.db_rows["messages.media_path"] == 1
    assert len(plan.avatars) == 1
    assert plan.avatars[0].action == mod.MOVE
    assert plan.executable


# ---------------------------------------------------------------------------
# No-conflict migration + idempotency
# ---------------------------------------------------------------------------


async def test_clean_rename_moves_folder_and_every_path_shape(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    (old / "sub").mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo-a")
    (old / "sub" / "b.bin").write_bytes(b"photo-b")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db,
        [-352],
        media_paths=["352/a.jpg", "/srv/media/352/sub/b.bin"],
        message_paths=["/srv/media/352/a.jpg"],
    )

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert not old.exists()
    new_dir = media / "-352"
    assert (new_dir / "a.jpg").read_bytes() == b"photo-a"
    assert (new_dir / "sub" / "b.bin").exists()

    engine = await _connect(db)
    try:
        media_paths, _, message_paths = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert sorted(media_paths) == ["-352/a.jpg", "/srv/media/-352/sub/b.bin"]
    assert message_paths == ["/srv/media/-352/a.jpg"]


async def test_rerun_after_success_is_a_noop(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo-a")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["352/a.jpg"])

    first = await mod.migrate(url, str(media), dry_run=False)
    second = await mod.migrate(url, str(media), dry_run=False)

    assert first.status == second.status == "applied"
    assert second.stats["folders_renamed"] == 0
    assert second.stats["entries_moved"] == 0
    assert second.stats["db_rows_updated"] == 0
    assert (media / "-352" / "a.jpg").exists()


async def test_already_migrated_archive_needs_nothing(tmp_path):
    media = tmp_path / "media"
    new_dir = media / "-352"
    new_dir.mkdir(parents=True)
    (new_dir / "a.jpg").write_bytes(b"photo-a")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["-352/a.jpg"])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert result.stats["folders_renamed"] == 0
    assert result.stats["db_rows_updated"] == 0
    assert (new_dir / "a.jpg").read_bytes() == b"photo-a"


# ---------------------------------------------------------------------------
# Same-name targets: prove equality before deleting anything
# ---------------------------------------------------------------------------


async def test_same_name_identical_bytes_are_deduplicated(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"same")
    (new_dir / "a.jpg").write_bytes(b"same")
    (old / "b.jpg").write_bytes(b"only-in-old")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db, [-352], media_paths=["352/a.jpg", "352/b.jpg"]
    )

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert not old.exists()
    assert (new_dir / "a.jpg").read_bytes() == b"same"
    assert (new_dir / "b.jpg").read_bytes() == b"only-in-old"
    assert result.stats["duplicates_removed"] == 1
    assert result.stats["entries_moved"] == 1


async def test_same_name_different_content_stops_before_any_change(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"old-bytes")
    (new_dir / "a.jpg").write_bytes(b"new-bytes")
    (old / "b.jpg").write_bytes(b"pending-move")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["352/a.jpg", "352/b.jpg"])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "conflicts"
    # Nothing moved, merged, removed or rewritten.
    assert (old / "a.jpg").read_bytes() == b"old-bytes"
    assert (new_dir / "a.jpg").read_bytes() == b"new-bytes"
    assert (old / "b.jpg").exists()
    engine = await _connect(db)
    try:
        media_paths, _, _ = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert sorted(media_paths) == ["352/a.jpg", "352/b.jpg"]


async def test_dry_run_reports_conflicts_without_writing(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"one")
    (new_dir / "a.jpg").write_bytes(b"two")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["352/a.jpg"])

    result = await mod.migrate(url, str(media), dry_run=True)

    assert result.status == "conflicts"
    assert (old / "a.jpg").read_bytes() == b"one"
    assert (new_dir / "a.jpg").read_bytes() == b"two"


async def test_type_mismatch_is_a_conflict(tmp_path):
    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "x").mkdir()
    (new_dir / "x").write_bytes(b"i am a file")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352])

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            plan = await mod.build_plan(conn, str(media))
    finally:
        await engine.dispose()

    assert not plan.executable
    conflict = plan.chats[0].relocations[0]
    assert conflict.action == mod.CONFLICT
    assert conflict.reason == mod.TYPE_MISMATCH


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


async def test_disk_renamed_but_db_stale_is_finished_on_rerun(tmp_path):
    """The exact half-migrated state the old commit order left behind."""
    media = tmp_path / "media"
    new_dir = media / "-352"
    new_dir.mkdir(parents=True)
    (new_dir / "a.jpg").write_bytes(b"photo-a")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db, [-352], media_paths=["352/a.jpg"], message_paths=["/srv/media/352/a.jpg"]
    )

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert result.stats["folders_renamed"] == 0
    assert result.stats["db_rows_updated"] == 2
    engine = await _connect(db)
    try:
        media_paths, _, message_paths = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert media_paths == ["-352/a.jpg"]
    assert message_paths == ["/srv/media/-352/a.jpg"]


async def test_resume_through_a_partial_merge_converges(tmp_path):
    """a.jpg is already at the target (identical), b.jpg never moved."""
    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"same")
    (new_dir / "a.jpg").write_bytes(b"same")
    (old / "b.jpg").write_bytes(b"b")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["352/a.jpg", "352/b.jpg"])

    result = await mod.migrate(url, str(media), dry_run=False)
    rerun = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert not old.exists()
    assert (new_dir / "a.jpg").read_bytes() == b"same"
    assert (new_dir / "b.jpg").read_bytes() == b"b"
    assert rerun.status == "applied"
    assert rerun.stats["folders_renamed"] == 0
    assert rerun.stats["db_rows_updated"] == 0


async def test_failed_stage_is_not_reported_complete_and_rerun_converges(tmp_path, monkeypatch):
    import os as _os

    media = tmp_path / "media"
    old = media / "352"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo-a")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-352], media_paths=["352/a.jpg"])

    real_rename = _os.rename
    calls = {"n": 0}

    def fail_first_rename(source, target):
        if calls["n"] == 0:
            calls["n"] += 1
            raise OSError("simulated kill mid-rename")
        return real_rename(source, target)

    monkeypatch.setattr(mod.os, "rename", fail_first_rename)
    failed = await mod.migrate(url, str(media), dry_run=False)
    monkeypatch.undo()

    assert failed.status == "failed"
    assert failed.error is not None
    assert old.exists() and (old / "a.jpg").exists()

    resumed = await mod.migrate(url, str(media), dry_run=False)
    assert resumed.status == "applied"
    assert not old.exists()
    assert (media / "-352" / "a.jpg").exists()
    engine = await _connect(db)
    try:
        media_paths, _, _ = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert media_paths == ["-352/a.jpg"]


# ---------------------------------------------------------------------------
# Avatars, including archives that have nothing else to migrate
# ---------------------------------------------------------------------------


async def test_avatar_only_old_archive(tmp_path):
    media = tmp_path / "media"
    avatar_dir = media / "avatars" / "chats"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "352_999.jpg").write_bytes(b"avatar")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert not (avatar_dir / "352_999.jpg").exists()
    assert (avatar_dir / "-352_999.jpg").read_bytes() == b"avatar"
    assert result.stats["avatars_moved"] == 1

    rerun = await mod.migrate(url, str(media), dry_run=False)
    assert rerun.status == "applied"
    assert rerun.stats["avatars_moved"] == 0
    assert rerun.stats["avatar_duplicates_removed"] == 0


async def test_identical_avatar_is_deduplicated(tmp_path):
    media = tmp_path / "media"
    avatar_dir = media / "avatars" / "chats"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "352_999.jpg").write_bytes(b"same")
    (avatar_dir / "-352_999.jpg").write_bytes(b"same")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert not (avatar_dir / "352_999.jpg").exists()
    assert (avatar_dir / "-352_999.jpg").read_bytes() == b"same"
    assert result.stats["avatar_duplicates_removed"] == 1


async def test_conflicting_avatar_blocks_the_whole_run(tmp_path):
    media = tmp_path / "media"
    avatar_dir = media / "avatars" / "chats"
    avatar_dir.mkdir(parents=True)
    (avatar_dir / "111_1.jpg").write_bytes(b"one")
    (avatar_dir / "-111_1.jpg").write_bytes(b"two")
    # A perfectly safe rename sits next to the conflict: it must wait too.
    (avatar_dir / "222_2.jpg").write_bytes(b"safe")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "conflicts"
    assert (avatar_dir / "111_1.jpg").read_bytes() == b"one"
    assert (avatar_dir / "-111_1.jpg").read_bytes() == b"two"
    assert (avatar_dir / "222_2.jpg").exists()
    assert not (avatar_dir / "-222_2.jpg").exists()


async def test_missing_avatars_directory_is_fine(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"


# ---------------------------------------------------------------------------
# Modern schema (messages.media_path gone, media_versions present)
# and PostgreSQL
# ---------------------------------------------------------------------------


async def test_modern_schema_uses_media_and_versions_and_skips_messages_column(real_db, tmp_path):
    media = tmp_path / "media"
    old = media / "456"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo")

    now = datetime(2026, 1, 1, 12, 0, 0)
    engine = real_db.engine
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO chats (account_id, id, ref, type, last_synced_message_id) "
                "VALUES (1, -456, 'refrefref1', 'group', 0)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO messages (account_id, id, chat_id, date, is_outgoing, is_pinned) "
                "VALUES (1, 1, -456, :d, 0, 0)"
            ),
            {"d": now},
        )
        await conn.execute(
            text(
                "INSERT INTO media (account_id, id, message_id, chat_id, file_path, downloaded) "
                "VALUES (1, 'm1', 1, -456, '456/a.jpg', 1)"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO media_versions (account_id, chat_id, message_id, media_id, downloaded, date, captured_at) "
                "VALUES (1, -456, 1, 'm1', 1, :d, :d)"
            ),
            {"d": now},
        )
        # Give the version row a path too.
        await conn.execute(
            text("UPDATE media_versions SET file_path = '456/old-version.bin'")
        )

    url = str(engine.url)
    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    async with engine.connect() as conn:
        media_path = (await conn.execute(text("SELECT file_path FROM media"))).scalar()
        version_path = (await conn.execute(text("SELECT file_path FROM media_versions"))).scalar()
    assert media_path == "-456/a.jpg"
    assert version_path == "-456/old-version.bin"
    assert (media / "-456" / "a.jpg").exists()
    assert not (media / "456").exists()


async def test_channel_folder_uses_unprefixed_legacy_name(tmp_path):
    """Channels lived in the bare positive folder, not 100+ prefixed (migration 013)."""
    media = tmp_path / "media"
    old = media / "456"
    old.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"photo")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(
        db, [-1_000_000_000_456], media_paths=["456/a.jpg"]
    )

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "applied"
    assert (media / "-1000000000456" / "a.jpg").exists()
    assert not old.exists()
    engine = await _connect(db)
    try:
        media_paths, _, _ = await _all_paths(engine)
    finally:
        await engine.dispose()
    assert media_paths == ["-1000000000456/a.jpg"]


async def test_two_chats_sharing_one_legacy_folder_blocks_execution(tmp_path):
    media = tmp_path / "media"
    shared = media / "456"
    shared.mkdir(parents=True)
    (shared / "a.jpg").write_bytes(b"photo")
    db = tmp_path / "archive.db"
    url = await _make_legacy_db(db, [-456, -1_000_000_000_456])

    result = await mod.migrate(url, str(media), dry_run=False)

    assert result.status == "conflicts"
    assert shared.exists() and (shared / "a.jpg").exists()
    assert not (media / "-456").exists()
    assert not (media / "-1000000000456").exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def test_cli_exits_nonzero_on_conflicts(tmp_path, monkeypatch):
    import asyncio

    media = tmp_path / "media"
    old = media / "352"
    new_dir = media / "-352"
    old.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old / "a.jpg").write_bytes(b"one")
    (new_dir / "a.jpg").write_bytes(b"two")
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'archive.db'}"
    await _make_legacy_db(tmp_path / "archive.db", [-352], media_paths=["352/a.jpg"])

    monkeypatch.setattr(
        "sys.argv",
        ["migrate_media_paths.py", "--dry-run", "--media-path", str(media), "--db-url", db_url],
    )
    # main() owns its own asyncio.run, so call it outside this test loop.
    with pytest.raises(SystemExit) as exc_info:
        await asyncio.to_thread(mod.main)
    assert exc_info.value.code == 2
    assert (old / "a.jpg").read_bytes() == b"one"
