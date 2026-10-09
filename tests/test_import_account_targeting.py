"""Imports target ONE resolved account, end to end.

The importer used to hard-code accounts.id 1, so in a multi-account archive an
import silently filed chats, messages, media, sync_status and its resume marker
under the wrong user. The target is now resolved ONCE, before any database or
filesystem write:

* a full JSON export auto-matches personal_information.user_id to an existing
  accounts row;
* ``--account`` (a stable accounts.id or a label) selects explicitly;
* an owner conflict, an unknown/ambiguous selection or a missing explicit
  choice in a multi-account install rejects the import with no side effects;
* HTML and single-chat exports cannot name an owner, so they require an
  explicit account when more than one account exists;
* a single-account install keeps working with no flags;
* the resume marker is bound to the account: a rerun can only continue the
  same account's progress and never touches another account's.

End-to-end cases run on a real engine via ``conftest.real_adapter`` (SQLite
always, PostgreSQL when reachable).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from telegram_archive.db.models import Account, Chat, Media, Message, SyncStatus, account_metadata_key
from telegram_archive.telegram_import import (
    ImportAccountError,
    TelegramImporter,
    peek_export_identity,
    resolve_import_account,
)

CHAT_A = 902001
CHAT_B_EXPORT = 902002
CHAT_B = -1000000902002
HTML_CHAT = -100200900

SAMPLE_HTML = """\
<html><body>
<div class="page_wrap">
 <div class="page_header"><div class="content"><div class="text bold">HTML Chat</div></div></div>
 <div class="page_body chat_page"><div class="history">
  <div class="message default clearfix" id="message100">
   <div class="body">
    <div class="pull_right date details" title="15.01.2024 10:00:00">10:00</div>
    <div class="from_name">Alice</div>
    <div class="text">Hello world!</div>
   </div>
  </div>
 </div></div>
</div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# Export builders
# --------------------------------------------------------------------------- #


def _msg(msg_id: int, text: str = "hola", *, from_id: str = "user9999", **extra) -> dict:
    base = {
        "id": msg_id,
        "type": "message",
        "date": f"2024-01-15T10:{msg_id % 60:02d}:00",
        "from": "Someone",
        "from_id": from_id,
        "text": text,
    }
    base.update(extra)
    return base


def _write_full_export(export_dir: Path, *, owner: int, chats: list[dict]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    control = export_dir / "result.json"
    control.write_text(
        json.dumps(
            {
                "about": "Telegram Desktop export",
                "personal_information": {"user_id": owner, "first_name": "Owner"},
                "chats": {"list": chats},
            }
        ),
        encoding="utf-8",
    )
    return control


def _write_single_chat_export(export_dir: Path, *, chat_id: int = CHAT_A) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    control = export_dir / "result.json"
    control.write_text(
        json.dumps(
            {
                "name": "Solo Chat",
                "type": "personal_chat",
                "id": chat_id,
                "messages": [_msg(1, "one"), _msg(2, "two", from_id="user4242")],
            }
        ),
        encoding="utf-8",
    )
    return control


def _write_html_export(export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    control = export_dir / "messages.html"
    control.write_text(SAMPLE_HTML, encoding="utf-8")
    return control


def _two_chats(export_dir: Path, *, owner: int = 9999, b_messages: int = 4, with_media: bool = False) -> Path:
    files_dir = export_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    chat_a = {
        "name": "Chat A",
        "type": "personal_chat",
        "id": CHAT_A,
        "messages": [_msg(1, "a1", from_id=f"user{owner}"), _msg(2, "a2")],
    }
    b_msgs = []
    for i in range(1, b_messages + 1):
        extra = {}
        if with_media and i == 2:
            (files_dir / "doc.bin").write_bytes(b"import-bytes")
            extra = {"file": "files/doc.bin", "file_name": "doc.bin", "media_type": "document"}
        b_msgs.append(_msg(i, f"b-{i}", from_id=f"user{owner}", **extra))
    chat_b = {"name": "Chat B", "type": "private_supergroup", "id": CHAT_B_EXPORT, "messages": b_msgs}
    return _write_full_export(export_dir, owner=owner, chats=[chat_a, chat_b])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _seed_accounts(adapter, *accounts: tuple[int, str | None, int | None]) -> None:
    async with adapter.db_manager.async_session_factory() as session:
        for account_id, label, telegram_user_id in accounts:
            session.add(Account(id=account_id, label=label, telegram_user_id=telegram_user_id))
        await session.commit()


def _importer(adapter, tmp_path: Path, **kwargs) -> TelegramImporter:
    return TelegramImporter(adapter, str(tmp_path / "media"), **kwargs)


async def _rows_for(adapter, model, account_id: int) -> list:
    async with adapter.db_manager.async_session_factory() as session:
        result = await session.execute(select(model).where(model.account_id == account_id))
        return list(result.scalars())


async def _state(adapter) -> dict:
    out = {}
    for account_id in (1, 2, 3):
        out[account_id] = {
            "chats": len(await _rows_for(adapter, Chat, account_id)),
            "messages": len(await _rows_for(adapter, Message, account_id)),
            "media": len(await _rows_for(adapter, Media, account_id)),
            "sync": len(await _rows_for(adapter, SyncStatus, account_id)),
        }
    return out


def _mock_db(rows: list[dict]) -> MagicMock:
    db = AsyncMock()
    db.get_account_identities = AsyncMock(return_value=rows)
    return db


# --------------------------------------------------------------------------- #
# Unit: resolve_import_account
# --------------------------------------------------------------------------- #


async def test_resolve_explicit_id_wins_when_owner_agrees():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    resolved = await resolve_import_account(db, selector=2, owner_user_id=9999)
    assert resolved == 2


async def test_resolve_numeric_string_selects_id():
    db = _mock_db([{"id": 1, "label": "default", "telegram_user_id": None}])
    assert await resolve_import_account(db, selector="1", owner_user_id=None) == 1


async def test_resolve_label_selects_the_account():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    assert await resolve_import_account(db, selector="work", owner_user_id=None) == 2


async def test_resolve_unknown_id_raises():
    db = _mock_db([{"id": 1, "label": "default", "telegram_user_id": 4242}])
    with pytest.raises(ImportAccountError, match="does not exist"):
        await resolve_import_account(db, selector=5, owner_user_id=None)


async def test_resolve_unknown_label_raises():
    db = _mock_db([{"id": 1, "label": "default", "telegram_user_id": 4242}])
    with pytest.raises(ImportAccountError, match="No account has the label"):
        await resolve_import_account(db, selector="ghost", owner_user_id=None)


async def test_resolve_duplicate_label_is_ambiguous():
    db = _mock_db(
        [
            {"id": 1, "label": "dup", "telegram_user_id": 1},
            {"id": 2, "label": "dup", "telegram_user_id": 2},
        ]
    )
    with pytest.raises(ImportAccountError, match="used by accounts 1, 2"):
        await resolve_import_account(db, selector="dup", owner_user_id=None)


async def test_resolve_owner_auto_matches_without_selector():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    assert await resolve_import_account(db, selector=None, owner_user_id=9999) == 2


async def test_resolve_owner_conflict_with_selector_raises():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    with pytest.raises(ImportAccountError, match="belongs to account 2"):
        await resolve_import_account(db, selector=1, owner_user_id=9999)


async def test_resolve_ambiguous_owner_raises():
    db = _mock_db(
        [
            {"id": 1, "label": "a", "telegram_user_id": 9999},
            {"id": 2, "label": "b", "telegram_user_id": 9999},
        ]
    )
    with pytest.raises(ImportAccountError, match="several accounts"):
        await resolve_import_account(db, selector=None, owner_user_id=9999)


async def test_resolve_ownerless_multi_account_requires_explicit():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    with pytest.raises(ImportAccountError, match="more than one account"):
        await resolve_import_account(db, selector=None, owner_user_id=None)


async def test_resolve_unknown_owner_multi_account_requires_explicit():
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
        ]
    )
    with pytest.raises(ImportAccountError, match="more than one account"):
        await resolve_import_account(db, selector=None, owner_user_id=77777)


async def test_resolve_single_account_legacy_picks_the_only_row():
    db = _mock_db([{"id": 3, "label": "solo", "telegram_user_id": None}])
    assert await resolve_import_account(db, selector=None, owner_user_id=None) == 3


async def test_resolve_no_account_rows_lands_on_default():
    assert await resolve_import_account(_mock_db([]), selector=None, owner_user_id=None) == 1


async def test_resolve_declared_count_makes_single_row_db_multi_account():
    # One row exists (account 2 has not logged in yet) but two are declared.
    db = _mock_db([{"id": 1, "label": "default", "telegram_user_id": None}])
    with pytest.raises(ImportAccountError, match="more than one account"):
        await resolve_import_account(db, selector=None, owner_user_id=None, configured_account_count=2)
    # ...but an explicit id still has to name an existing row.
    with pytest.raises(ImportAccountError, match="does not exist"):
        await resolve_import_account(db, selector=2, owner_user_id=None, configured_account_count=2)


async def test_resolve_selector_allows_unowned_row_when_owner_unknown():
    # Row 3 has never logged in (telegram_user_id NULL); the export's owner is
    # not any known account, so selecting row 3 is an explicit override, not a
    # conflict with a different KNOWN owner.
    db = _mock_db(
        [
            {"id": 1, "label": "default", "telegram_user_id": 4242},
            {"id": 2, "label": "work", "telegram_user_id": 9999},
            {"id": 3, "label": "spare", "telegram_user_id": None},
        ]
    )
    assert await resolve_import_account(db, selector=3, owner_user_id=55555) == 3


# --------------------------------------------------------------------------- #
# Unit: peek_export_identity (read-only)
# --------------------------------------------------------------------------- #


async def test_peek_full_export_reports_owner(tmp_path):
    control = _two_chats(tmp_path / "export")
    assert peek_export_identity(control) == ("full", 9999)


async def test_peek_single_chat_export_has_no_owner(tmp_path):
    control = _write_single_chat_export(tmp_path / "export")
    assert peek_export_identity(control) == ("single", None)


async def test_peek_full_export_without_personal_information(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    control = export_dir / "result.json"
    control.write_text(
        json.dumps({"chats": {"list": [{"name": "x", "type": "personal_chat", "id": 1, "messages": []}]}})
    )
    assert peek_export_identity(control) == ("full", None)


# --------------------------------------------------------------------------- #
# End to end on a real engine
# --------------------------------------------------------------------------- #


async def test_full_json_auto_matches_owner_account(real_adapter, tmp_path):
    await _seed_accounts(
        real_adapter,
        (1, "default", 4242),
        (2, "work", 9999),
    )
    _two_chats(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))

    assert summary["account_id"] == 2
    assert {detail["account_id"] for detail in summary["details"]} == {2}

    state = await _state(real_adapter)
    assert state[2]["chats"] == 2
    assert state[2]["messages"] == 6
    assert state[2]["sync"] == 2
    # Account 1 is completely untouched: chats/messages/media/sync_status.
    assert state[1] == {"chats": 0, "messages": 0, "media": 0, "sync": 0}


async def test_full_json_media_row_and_file_land_with_owner_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export", with_media=True)
    await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))

    media = await _rows_for(real_adapter, Media, 2)
    assert len(media) == 1
    assert media[0].id == f"import_{CHAT_B}_2"
    # The copied bytes live in the shared media store under that chat.
    assert (tmp_path / "media" / str(CHAT_B)).exists()
    assert any((tmp_path / "media" / str(CHAT_B)).iterdir())
    assert await _rows_for(real_adapter, Media, 1) == []


async def test_explicit_label_targets_the_chosen_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _write_single_chat_export(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path, account="work").run(str(tmp_path / "export"))
    assert summary["account_id"] == 2
    assert len(await _rows_for(real_adapter, Message, 2)) == 2
    assert await _rows_for(real_adapter, Message, 1) == []


async def test_conflicting_explicit_account_rejects_without_writes(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export", with_media=True)

    with pytest.raises(ImportAccountError, match="belongs to account 2"):
        await _importer(real_adapter, tmp_path, account=1).run(str(tmp_path / "export"))

    # Zero rows under EITHER account, no resume marker, and no media directory
    # created: the refusal happened before the first write.
    state = await _state(real_adapter)
    assert state[1] == {"chats": 0, "messages": 0, "media": 0, "sync": 0}
    assert state[2] == {"chats": 0, "messages": 0, "media": 0, "sync": 0}
    assert not (tmp_path / "media").exists()
    for account_id in (1, 2):
        key = account_metadata_key("import_progress", account_id)
        assert not await real_adapter.get_metadata(key)


async def test_nonexistent_account_rejects_without_writes(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export")
    with pytest.raises(ImportAccountError, match="does not exist"):
        await _importer(real_adapter, tmp_path, account=9).run(str(tmp_path / "export"))
    assert await _rows_for(real_adapter, Message, 1) == []
    assert await _rows_for(real_adapter, Message, 2) == []


async def test_html_multi_account_requires_explicit_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _write_html_export(tmp_path / "export")
    with pytest.raises(ImportAccountError, match="more than one account"):
        await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"), chat_id_override=HTML_CHAT)
    assert await _rows_for(real_adapter, Chat, 1) == []
    assert await _rows_for(real_adapter, Chat, 2) == []


async def test_html_multi_account_succeeds_with_explicit_id(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _write_html_export(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path, account=2).run(
        str(tmp_path / "export"), chat_id_override=HTML_CHAT
    )
    assert summary["account_id"] == 2
    assert len(await _rows_for(real_adapter, Message, 2)) == 1
    assert await _rows_for(real_adapter, Message, 1) == []


async def test_single_chat_json_multi_account_requires_explicit_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _write_single_chat_export(tmp_path / "export")
    with pytest.raises(ImportAccountError, match="more than one account"):
        await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))


async def test_html_single_account_legacy_needs_no_flag(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242))
    _write_html_export(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"), chat_id_override=HTML_CHAT)
    assert summary["account_id"] == 1
    assert len(await _rows_for(real_adapter, Message, 1)) == 1


async def test_single_chat_json_single_account_legacy_needs_no_flag(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242))
    _write_single_chat_export(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))
    assert summary["account_id"] == 1
    assert len(await _rows_for(real_adapter, Message, 1)) == 2


async def test_full_json_unknown_owner_multi_account_must_choose(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export", owner=77777)
    with pytest.raises(ImportAccountError, match="more than one account"):
        await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))
    # Explicit selection into an existing account is then honored.
    summary = await _importer(real_adapter, tmp_path, account=2).run(str(tmp_path / "export"))
    assert summary["account_id"] == 2
    assert len(await _rows_for(real_adapter, Message, 2)) == 6


async def test_declared_multi_account_with_one_row_still_requires_choice(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242))
    _write_html_export(tmp_path / "export")
    with pytest.raises(ImportAccountError, match="more than one account"):
        await _importer(real_adapter, tmp_path, configured_account_count=2).run(
            str(tmp_path / "export"), chat_id_override=HTML_CHAT
        )
    summary = await _importer(real_adapter, tmp_path, account=1, configured_account_count=2).run(
        str(tmp_path / "export"), chat_id_override=HTML_CHAT
    )
    assert summary["account_id"] == 1


async def test_dry_run_resolves_owner_but_writes_nothing(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export")
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"), dry_run=True)
    assert summary["account_id"] == 2
    assert summary["total_messages"] == 6
    state = await _state(real_adapter)
    assert state[2]["messages"] == 0
    assert not (tmp_path / "media").exists()


async def test_dry_run_still_rejects_ownerless_multi_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _write_html_export(tmp_path / "export")
    with pytest.raises(ImportAccountError, match="more than one account"):
        await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"), chat_id_override=HTML_CHAT, dry_run=True)


async def test_interrupted_import_resumes_under_the_same_account(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    _two_chats(tmp_path / "export")

    real_batch = real_adapter.insert_messages_batch
    calls = {"n": 0}

    async def explode_on_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("interrupt: simulated crash mid-import")
        return await real_batch(*args, **kwargs)

    real_adapter.insert_messages_batch = explode_on_third
    try:
        with (
            patch("telegram_archive.telegram_import.BATCH_SIZE", 2),
            pytest.raises(RuntimeError, match="simulated crash"),
        ):
            # No selector: the owner (9999) auto-resolves to account 2.
            await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))
    finally:
        real_adapter.insert_messages_batch = real_batch

    marker_key = account_metadata_key("import_progress", 2)
    marker = json.loads(await real_adapter.get_metadata(marker_key))
    assert marker["account_id"] == 2
    assert marker["completed"] == [CHAT_A]

    # A FRESH process re-runs with no selector: the owner resolves to account 2
    # again and only its own marker is consumed.
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))
    assert summary["chats_skipped"] == 1
    assert summary["chats_imported"] == 1

    assert len(await _rows_for(real_adapter, Message, 2)) == 6
    assert await _rows_for(real_adapter, Message, 1) == []
    sync = await _rows_for(real_adapter, SyncStatus, 2)
    assert {(row.chat_id, row.last_message_id, row.message_count) for row in sync} == {
        (CHAT_A, 2, 2),
        (CHAT_B, 4, 4),
    }
    # Clean completion clears account 2's marker and never wrote account 1's.
    assert not await real_adapter.get_metadata(marker_key)
    assert not await real_adapter.get_metadata(account_metadata_key("import_progress", 1))


async def test_marker_naming_another_account_is_refused(real_adapter, tmp_path):
    """The embedded account id guards the per-account key itself: a marker that
    claims another account's progress must not be continued or cleared."""
    await _seed_accounts(real_adapter, (1, "default", 4242), (2, "work", 9999))
    control = _write_single_chat_export(tmp_path / "export")
    from telegram_archive.telegram_import import _export_fingerprint

    # Tampered/colliding marker on account 1's own key, claiming account 2.
    await real_adapter.set_metadata(
        account_metadata_key("import_progress", 1),
        json.dumps({"fingerprint": _export_fingerprint(control), "completed": [], "started": CHAT_A, "account_id": 2}),
    )
    with pytest.raises(ImportAccountError, match="belongs to account 2"):
        await _importer(real_adapter, tmp_path, account=1).run(str(tmp_path / "export"))
    # Nothing was written and the foreign marker is left exactly as found.
    assert await _rows_for(real_adapter, Message, 1) == []
    raw = await real_adapter.get_metadata(account_metadata_key("import_progress", 1))
    assert json.loads(raw)["account_id"] == 2


async def test_legacy_marker_without_account_field_still_resumes_account_one(real_adapter, tmp_path):
    await _seed_accounts(real_adapter, (1, "default", 4242))
    control = _write_single_chat_export(tmp_path / "export")
    from telegram_archive.telegram_import import _export_fingerprint

    await real_adapter.set_metadata(
        account_metadata_key("import_progress", 1),
        json.dumps({"fingerprint": _export_fingerprint(control), "completed": [], "started": None}),
    )
    # A legacy marker (no account_id) is accepted for the default account and
    # immediately superseded by an account-bound one on the next checkpoint.
    summary = await _importer(real_adapter, tmp_path).run(str(tmp_path / "export"))
    assert summary["account_id"] == 1
    assert len(await _rows_for(real_adapter, Message, 1)) == 2


# --------------------------------------------------------------------------- #
# CLI: run_import threads --account through
# --------------------------------------------------------------------------- #


async def test_run_import_passes_account_selector(monkeypatch):
    from telegram_archive.__main__ import run_import

    mock_config = MagicMock()
    mock_config.media_path = "/data/media"
    mock_config.max_filename_bytes = 143
    mock_config.accounts = [MagicMock(), MagicMock()]

    mock_importer = AsyncMock()
    mock_importer.run = AsyncMock(
        return_value={
            "account_id": 2,
            "chats_imported": 1,
            "total_messages": 2,
            "total_media": 0,
            "details": [{"account_id": 2, "chat_name": "Solo Chat", "chat_id": CHAT_A, "messages": 2, "media": 0}],
        }
    )
    mock_importer.close = AsyncMock()
    create_mock = AsyncMock(return_value=mock_importer)

    args = SimpleNamespace(
        path="/tmp/export",
        chat_id=None,
        dry_run=True,
        skip_media=True,
        merge=False,
        account="work",
    )
    with (
        patch("telegram_archive.config.Config", return_value=mock_config),
        patch("telegram_archive.config.setup_logging"),
        patch("telegram_archive.telegram_import.TelegramImporter.create", create_mock),
    ):
        result = await run_import(args)

    assert result == 0
    assert create_mock.await_args.kwargs["account"] == "work"
    assert create_mock.await_args.kwargs["configured_account_count"] == 2
    mock_importer.run.assert_awaited_once()


# --------------------------------------------------------------------------- #
# Merge carries the marker's embedded account to the remapped target account
# --------------------------------------------------------------------------- #


def test_merge_remap_helper_rewrites_only_marker_account_id():
    from telegram_archive.merge import _remap_import_marker_value

    marker = json.dumps({"fingerprint": "1:aa", "completed": [7], "started": None, "account_id": 2})
    remapped = json.loads(_remap_import_marker_value(marker, 3))
    assert remapped == {"fingerprint": "1:aa", "completed": [7], "started": None, "account_id": 3}
    # A legacy bare-text value and JSON without the field pass through untouched.
    assert _remap_import_marker_value("fake progress", 3) == "fake progress"
    other = json.dumps({"fingerprint": "1:aa", "completed": [7]})
    assert _remap_import_marker_value(other, 3) == other


def test_merge_metadata_rows_rekey_and_rewrite_markers(tmp_path):
    import sqlalchemy as sa

    from telegram_archive.merge import account_metadata_rows

    engine = sa.create_engine(f"sqlite:///{tmp_path / 'merge_meta.db'}")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value TEXT)"))
        conn.execute(
            sa.text("INSERT INTO metadata (key, value) VALUES ('import_progress_account_2', :v)"),
            {"v": json.dumps({"fingerprint": "1:aa", "completed": [7], "started": None, "account_id": 2})},
        )
        conn.execute(sa.text("INSERT INTO metadata (key, value) VALUES ('import_progress', 'fake progress')"))
        conn.execute(sa.text("INSERT INTO metadata (key, value) VALUES ('some_global_key', 'x')"))

    with engine.connect() as conn:
        rows = account_metadata_rows(conn, {1: 5, 2: 3})

    by_key = {row["key"]: row["value"] for row in rows}
    assert set(by_key) == {"import_progress_account_3", "import_progress_account_5"}
    assert json.loads(by_key["import_progress_account_3"])["account_id"] == 3
    assert by_key["import_progress_account_5"] == "fake progress"  # legacy value byte-identical
    engine.dispose()
