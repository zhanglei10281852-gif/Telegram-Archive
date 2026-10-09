"""``scripts/restore_chat.py`` resumes: a run is a durable, idempotent job.

The job state file binds the source chat, destination chat, filters, the media
order and the confirmed Telegram result of every text/caption/file send. A
re-run with the same arguments sends only units that are not confirmed, so an
interrupted run converges to exactly what one successful run would have sent.
A send whose delivery is unknown blocks the job in a diagnosable state rather
than risking a duplicate.

Since 9.0 the export lists a message once, with all its media rows. The
restore sends the first file on disk with the text as its caption and every
other file after it without text, in the order the export lists media:
downloaded first, then by the lowest id. Runs on SQLite and PostgreSQL
(``real_adapter``).
"""

import glob
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telethon.errors import (
    ChatWriteForbiddenError,
    FloodWaitError,
    SlowModeWaitError,
)

CHAT = -1001900000002
SENT = datetime(2026, 10, 1, 10, 38, 0)
REPO = Path(__file__).resolve().parents[1]

CAPTION = "[Fixture Sender - 2026-10-01 10:38]\nThree photos from the trip"
TEXT_OF_2 = "[Fixture Sender - 2026-10-01 10:38]\nOnly text"


def _msg(message_id: int):
    """What Telethon returns for a successful send: a message with an id."""
    return SimpleNamespace(id=message_id, date=SENT)


def _load_restore():
    spec = importlib.util.spec_from_file_location("restore_chat", REPO / "scripts" / "restore_chat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _album(adapter, media_root: Path) -> None:
    """Message 1 holds four media rows; message 2 is text only.

    ``fixture-0`` is downloaded but its file is gone, ``fixture-a`` was never
    downloaded, and ``fixture-c`` and ``fixture-b`` are on disk (inserted in
    that order, so the order is the id's, not the insert's).
    """
    await adapter.upsert_chat({"id": CHAT, "type": "group", "title": "Fixture Group"}, account_id=1)
    for message_id, text in ((1, "Three photos from the trip"), (2, "Only text")):
        await adapter.insert_message(
            {"id": message_id, "chat_id": CHAT, "text": text, "date": SENT, "sender_name": "Fixture Sender"},
            account_id=1,
        )
    folder = media_root / str(CHAT)
    folder.mkdir(parents=True)
    for media_id, downloaded, on_disk in (
        ("fixture-c", True, True),
        ("fixture-b", True, True),
        ("fixture-0", True, False),
        ("fixture-a", False, False),
    ):
        path = f"{CHAT}/{media_id}.jpg" if downloaded else None
        if on_disk:
            (folder / f"{media_id}.jpg").write_bytes(b"demo photo " + media_id.encode())
        await adapter.insert_media(
            {
                "id": media_id,
                "message_id": 1,
                "chat_id": CHAT,
                "type": "photo",
                "file_path": path,
                "downloaded": downloaded,
            },
            account_id=1,
        )


async def _text_chat(adapter, count: int) -> None:
    """A chat of ``count`` text-only messages, ids 1..count, oldest first."""
    await adapter.upsert_chat({"id": CHAT, "type": "group", "title": "Fixture Group"}, account_id=1)
    for message_id in range(1, count + 1):
        await adapter.insert_message(
            {
                "id": message_id,
                "chat_id": CHAT,
                "text": f"note {message_id}",
                "date": SENT,
                "sender_name": "Fixture Sender",
            },
            account_id=1,
        )


async def test_the_adapter_hands_the_restore_every_media_row_in_export_order(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")

    restore = {m["id"]: m async for m in real_adapter.get_messages_for_export(CHAT, include_media=True, account_id=1)}
    plain = {m["id"]: m async for m in real_adapter.get_messages_for_export(CHAT, account_id=1)}

    assert [(f["type"], f["path"]) for f in restore[1]["media_files"]] == [
        ("photo", f"{CHAT}/fixture-0.jpg"),
        ("photo", f"{CHAT}/fixture-b.jpg"),
        ("photo", f"{CHAT}/fixture-c.jpg"),
        ("photo", None),
    ]
    assert [m["media_id"] for m in restore[1]["media"]] == ["fixture-0", "fixture-b", "fixture-c", "fixture-a"]
    assert restore[2]["media_files"] == []
    # The viewer's export never carries a path.
    assert all("media_files" not in m and "media_path" not in m for m in plain.values())


def _ok_client(send_file=None, send_message=None, get_entity=None):
    return SimpleNamespace(
        get_entity=get_entity or AsyncMock(return_value=SimpleNamespace(title="Fixture Destination")),
        send_file=send_file if send_file is not None else _auto_send(),
        send_message=send_message if send_message is not None else _auto_send(),
        disconnect=AsyncMock(),
    )


def _auto_send():
    counter = {"n": 100}

    async def fake(*args, **kwargs):
        counter["n"] += 1
        return _msg(counter["n"])

    return AsyncMock(side_effect=fake)


def _file_sends(client) -> list[tuple[str, str | None]]:
    return [(os.path.basename(call.args[1]), call.kwargs.get("caption")) for call in client.send_file.await_args_list]


def _text_sends(client) -> list[str]:
    return [call.args[1] for call in client.send_message.await_args_list]


async def _run(
    adapter,
    tmp_path,
    *,
    client=None,
    client_factory=None,
    job_dir=None,
    answer="YES",
    send_file=None,
    send_message=None,
    get_entity=None,
    **kwargs,
):
    """Run one restore into a fake client; returns outcome, client, module, job_dir."""
    module = _load_restore()
    client = client or _ok_client(send_file=send_file, send_message=send_message, get_entity=get_entity)
    factory = client_factory or AsyncMock(return_value=client)
    job_dir = job_dir or str(tmp_path / "jobs")
    with (
        patch.object(module, "get_db_adapter", AsyncMock(return_value=adapter)),
        patch.object(module, "get_telegram_client", factory),
        patch("builtins.input", return_value=answer),
        patch.object(module.asyncio, "sleep", new=AsyncMock()),
        patch.dict(os.environ, {"BACKUP_PATH": str(tmp_path)}),
    ):
        outcome = await module.restore_chat(CHAT, CHAT, delay=0, job_dir=job_dir, **kwargs)
    return outcome, client, module, job_dir


def _job_on_disk(job_dir: str) -> tuple[dict, str]:
    files = glob.glob(os.path.join(job_dir, "restore-job-*.json"))
    assert len(files) == 1, files
    with open(files[0], encoding="utf-8") as fh:
        return json.load(fh), files[0]


def _set_job_state(job_dir: str, message_id: int, key: str, state: str, *, clear_blocked=False) -> dict:
    """Hand-edit a job file (e.g. simulate a process killed mid-send)."""
    data, path = _job_on_disk(job_dir)
    for item in data["items"]:
        if item["message_id"] == message_id:
            for unit in item["units"]:
                if unit["key"] == key:
                    unit["state"] = state
    if clear_blocked:
        data["status"] = "active"
        data["halt_reason"] = None
        data["blocked"] = None
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


# ---------------------------------------------------------------------------
# The original send semantics: one message, every file, text once, waits retried
# ---------------------------------------------------------------------------


async def test_a_message_is_sent_once_with_every_file_it_has_on_disk(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")

    outcome, client, _module, job_dir = await _run(real_adapter, tmp_path)

    assert client.send_file.await_args_list[0].args[1] == str(
        tmp_path / "media" / str(CHAT) / "fixture-b.jpg"
    )
    assert _file_sends(client) == [("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None)]
    assert _text_sends(client) == [TEXT_OF_2]
    assert outcome.status == "complete"
    data, _ = _job_on_disk(job_dir)
    assert data["status"] == "complete"
    assert [u["state"] for u in data["items"][0]["units"]] == ["sent", "sent"]
    assert data["items"][0]["units"][0]["dest_message_id"] > 0


async def test_a_wait_on_a_later_file_sends_that_file_again_and_the_rest_after_it(real_adapter, tmp_path):
    """Telegram makes the second upload wait once: the same file goes again
    after the wait, and the message is not left with one of its files."""
    await _album(real_adapter, tmp_path / "media")
    send_file = AsyncMock(side_effect=[_msg(1), FloodWaitError(request=None, capture=0), _msg(2)])

    _outcome, client, _m, _jd = await _run(real_adapter, tmp_path, send_file=send_file)

    assert _file_sends(client) == [("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None), ("fixture-c.jpg", None)]
    assert client.send_message.await_count == 1


async def test_a_wait_on_the_first_file_or_a_text_sends_it_again_once(real_adapter, tmp_path):
    """The first file carries the text: after a wait it goes again with its
    caption, so the text is still sent exactly once, and so is a text-only message."""
    await _album(real_adapter, tmp_path / "media")
    send_file = AsyncMock(side_effect=[SlowModeWaitError(request=None, capture=0), _msg(1), _msg(2)])
    send_message = AsyncMock(
        side_effect=[FloodWaitError(request=None, capture=0), _msg(3)]
    )

    _outcome, client, _m, _jd = await _run(real_adapter, tmp_path, send_file=send_file, send_message=send_message)

    assert _file_sends(client) == [("fixture-b.jpg", CAPTION), ("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None)]
    assert _text_sends(client) == [TEXT_OF_2, TEXT_OF_2]


async def test_a_wait_that_never_clears_leaves_the_unit_pending_and_run_is_not_complete(
    real_adapter, tmp_path, caplog
):
    """After all wait retries the file is unconfirmed (state pending) and the
    run walks the next message; the job is not reported complete and a re-run
    retries the file instead of resending the caption."""
    await _album(real_adapter, tmp_path / "media")
    retries = _load_restore().SEND_WAIT_RETRIES
    waits = [FloodWaitError(request=None, capture=0)] * (retries + 1)
    send_file = AsyncMock(side_effect=[_msg(1), *waits])

    outcome, client, _m, job_dir = await _run(real_adapter, tmp_path, send_file=send_file)

    assert send_file.await_count == 1 + retries + 1
    assert _file_sends(client) == [
        ("fixture-b.jpg", CAPTION),
        ("fixture-c.jpg", None),
        ("fixture-c.jpg", None),
        ("fixture-c.jpg", None),
        ("fixture-c.jpg", None),
    ]
    # The next message still goes out.
    assert client.send_message.await_count == 1
    assert outcome.status == "active"
    assert outcome.halt_reason == "errors_remaining"
    assert outcome.pending == 1
    assert not any("RESTORE COMPLETE" in r.getMessage() for r in caplog.records)
    data, _ = _job_on_disk(job_dir)
    states = {(u["key"]): u["state"] for u in data["items"][0]["units"]}
    assert states == {"fixture-b": "sent", "fixture-c": "pending"}


# ---------------------------------------------------------------------------
# Resumable job: only unconfirmed units go out again
# ---------------------------------------------------------------------------


async def test_an_ambiguous_network_failure_blocks_and_resume_sends_only_that_file(
    real_adapter, tmp_path, caplog
):
    await _album(real_adapter, tmp_path / "media")

    # Run 1: the caption file is confirmed, the second upload loses the network.
    send_file_1 = AsyncMock(side_effect=[_msg(101), ConnectionError("connection reset")])
    run1, client1, module, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file_1, send_message=_auto_send()
    )

    assert run1.status == "blocked"
    assert run1.halt_reason == "unconfirmed_send"
    assert _file_sends(client1) == [("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None)]
    assert client1.send_message.await_count == 0  # later messages never ran
    client1.disconnect.assert_awaited_once()
    assert not any("RESTORE COMPLETE" in r.getMessage() for r in caplog.records)
    data, path = _job_on_disk(job_dir)
    assert data["status"] == "blocked"
    assert data["blocked"]["message_id"] == 1
    assert data["blocked"]["unit_key"] == "fixture-c"
    assert "ConnectionError" in data["blocked"]["error"]

    # Run 2 with no adjudication: the job stays blocked without even connecting.
    factory2 = AsyncMock(side_effect=AssertionError("must not connect"))
    run2, _client2, _m, _jd = await _run(
        real_adapter, tmp_path, client_factory=factory2, job_dir=job_dir
    )
    assert run2.status == "blocked"
    factory2.assert_not_awaited()

    # Run 3 after checking the chat: the missing file really did not arrive.
    send_file_3 = AsyncMock(side_effect=[_msg(201)])
    run3, client3, _m, _jd = await _run(
        real_adapter, tmp_path,
        send_file=send_file_3, send_message=_auto_send(),
        job_dir=job_dir, resend_unconfirmed=True,
    )

    assert run3.status == "complete"
    # Only the unconfirmed file is re-sent: the caption and the first file are
    # not repeated, and the file still carries no caption.
    assert _file_sends(client3) == [("fixture-c.jpg", None)]
    assert _text_sends(client3) == [TEXT_OF_2]
    # Across every run the chat received exactly the one successful-run result.
    assert [s for sends in (_file_sends(client1), _file_sends(client3)) for s in sends] == [
        ("fixture-b.jpg", CAPTION),
        ("fixture-c.jpg", None),
        ("fixture-c.jpg", None),
    ]
    data, _ = _job_on_disk(job_dir)
    assert data["status"] == "complete"


async def test_mark_unconfirmed_sent_finishes_without_resending(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")

    send_file_1 = AsyncMock(side_effect=[_msg(101), ConnectionError("gone")])
    run1, _client1, _m, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file_1, send_message=_auto_send()
    )
    assert run1.status == "blocked"

    # Operator checked the destination: the second file IS there.
    send_file_2 = AsyncMock(side_effect=AssertionError("must not send anything"))
    run2, client2, _m, _jd = await _run(
        real_adapter, tmp_path,
        send_file=send_file_2, send_message=_auto_send(),
        job_dir=job_dir, mark_unconfirmed_sent=True,
    )

    assert run2.status == "complete"
    send_file_2.assert_not_awaited()
    assert _text_sends(client2) == [TEXT_OF_2]
    data, _ = _job_on_disk(job_dir)
    c = next(u for u in data["items"][0]["units"] if u["key"] == "fixture-c")
    assert c["state"] == "sent"
    assert c["dest_message_id"] is None
    assert "operator confirmed" in c["note"]


async def test_a_process_killed_mid_send_is_ambiguous_on_the_next_run(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")

    send_file = AsyncMock(side_effect=[_msg(101), ConnectionError("gone")])
    run1, _c, _m, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file, send_message=_auto_send()
    )
    assert run1.status == "blocked"

    # Simulate a hard kill while the second upload's call was in flight: the
    # durable marker is "sending" and no result ever came back.
    _set_job_state(job_dir, 1, "fixture-c", "sending", clear_blocked=True)

    factory2 = AsyncMock(side_effect=AssertionError("must not connect"))
    run2, _c2, _m2, _jd = await _run(real_adapter, tmp_path, client_factory=factory2, job_dir=job_dir)
    assert run2.status == "blocked"
    factory2.assert_not_awaited()
    data, _ = _job_on_disk(job_dir)
    assert data["blocked"]["unit_key"] == "fixture-c"
    assert "in flight" in data["blocked"]["error"]

    run3, client3, _m3, _jd = await _run(
        real_adapter, tmp_path,
        send_file=AsyncMock(side_effect=[_msg(301)]), send_message=_auto_send(),
        job_dir=job_dir, resend_unconfirmed=True,
    )
    assert run3.status == "complete"
    assert _file_sends(client3) == [("fixture-c.jpg", None)]


async def test_keyboard_interrupt_saves_progress_disconnects_and_never_completes(
    real_adapter, tmp_path, caplog
):
    await _album(real_adapter, tmp_path / "media")
    send_file = AsyncMock(side_effect=[_msg(101), KeyboardInterrupt()])

    run1, client1, _m, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file, send_message=_auto_send()
    )

    assert run1.status == "active"
    assert run1.halt_reason == "interrupted"
    client1.disconnect.assert_awaited_once()
    assert not any("RESTORE COMPLETE" in r.getMessage() for r in caplog.records)
    data, _ = _job_on_disk(job_dir)
    states = {u["key"]: u["state"] for u in data["items"][0]["units"]}
    assert states == {"fixture-b": "sent", "fixture-c": "sending"}

    # Next run treats the interrupted call as ambiguous; once adjudicated the
    # job converges with the caption never sent twice.
    run2, client2, _m2, _jd = await _run(
        real_adapter, tmp_path,
        send_file=AsyncMock(side_effect=[_msg(401)]), send_message=_auto_send(),
        job_dir=job_dir, resend_unconfirmed=True,
    )
    assert run2.status == "complete"
    assert _file_sends(client2) == [("fixture-c.jpg", None)]
    assert _text_sends(client2) == [TEXT_OF_2]


async def test_error_threshold_stops_persists_and_a_retry_converges(real_adapter, tmp_path, caplog):
    await _text_chat(real_adapter, 22)

    refused = AsyncMock(side_effect=lambda *a, **k: (_ for _ in ()).throw(ChatWriteForbiddenError(request=None)))
    run1, client1, _m, job_dir = await _run(real_adapter, tmp_path, send_message=refused)

    assert run1.status == "active"
    assert run1.halt_reason == "error_threshold"
    assert run1.errors == 21  # stops as the 21st error is recorded
    assert client1.send_message.await_count == 21  # the 22nd message was never attempted
    client1.disconnect.assert_awaited_once()
    assert not any("RESTORE COMPLETE" in r.getMessage() for r in caplog.records)
    data, _ = _job_on_disk(job_dir)
    assert data["halt_reason"] == "error_threshold"
    assert all(u["state"] == "pending" for item in data["items"] for u in item["units"])

    # Permission fixed: the same job now sends every message exactly once.
    run2, client2, _m2, _jd = await _run(
        real_adapter, tmp_path, send_message=_auto_send(), job_dir=job_dir
    )
    assert run2.status == "complete"
    assert client2.send_message.await_count == 22
    assert _text_sends(client2) == [
        f"[Fixture Sender - 2026-10-01 10:38]\nnote {i}" for i in range(1, 23)
    ]


async def test_target_inaccessible_saves_progress_disconnects_and_resumes(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")
    bad_entity = AsyncMock(side_effect=ValueError("could not find the chat"))
    client1 = _ok_client(get_entity=bad_entity, send_file=_auto_send())
    factory1 = AsyncMock(return_value=client1)

    run1, client1, _m, job_dir = await _run(
        real_adapter, tmp_path, client=client1, client_factory=factory1
    )

    assert run1.status == "active"
    assert run1.halt_reason == "target_inaccessible"
    client1.disconnect.assert_awaited_once()
    client1.send_file.assert_not_awaited()
    data, _ = _job_on_disk(job_dir)
    assert all(u["state"] == "pending" for item in data["items"] for u in item["units"])

    run2, client2, _m2, _jd = await _run(real_adapter, tmp_path, job_dir=job_dir)
    assert run2.status == "complete"
    assert _file_sends(client2) == [("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None)]


async def test_connect_failure_saves_the_job_without_reporting_complete(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")
    factory = AsyncMock(side_effect=ConnectionRefusedError("Telegram unreachable"))

    run1, _c, _m, job_dir = await _run(real_adapter, tmp_path, client_factory=factory)

    assert run1.status == "active"
    assert run1.halt_reason == "connect_failed"
    factory.assert_awaited_once()
    data, path = _job_on_disk(job_dir)
    assert data["status"] == "active"
    assert data["halt_reason"] == "connect_failed"

    run2, client2, _m2, _jd = await _run(real_adapter, tmp_path, job_dir=job_dir)
    assert run2.status == "complete"
    assert _file_sends(client2) == [("fixture-b.jpg", CAPTION), ("fixture-c.jpg", None)]


async def test_a_reply_without_a_message_id_is_unconfirmed(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")
    send_file = AsyncMock(side_effect=[SimpleNamespace(id=None, date=None)])

    run1, _c, _m, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file, send_message=_auto_send()
    )

    assert run1.status == "blocked"
    data, _ = _job_on_disk(job_dir)
    assert data["blocked"]["unit_key"] == "fixture-b"
    assert data["blocked"]["caption"] is True
    assert "no message id" in data["blocked"]["error"]


# ---------------------------------------------------------------------------
# Job identity, dry run, drift, idempotence, filters
# ---------------------------------------------------------------------------


async def test_dry_run_uses_the_plan_but_writes_nothing(real_adapter, tmp_path, caplog):
    await _album(real_adapter, tmp_path / "media")
    job_dir = str(tmp_path / "jobs")
    factory = AsyncMock(side_effect=AssertionError("dry run never connects"))

    with caplog.at_level("INFO"):
        run1, _c, module, _jd = await _run(
            real_adapter, tmp_path, client_factory=factory, job_dir=job_dir, dry_run=True
        )

    assert run1.status == "dry_run"
    factory.assert_not_awaited()
    assert not os.path.exists(job_dir)
    assert any("DRY RUN PREVIEW" in r.getMessage() for r in caplog.records)
    assert any("MISSING" in r.getMessage() for r in caplog.records)


async def test_the_same_arguments_resume_one_job_other_filters_make_another(real_adapter, tmp_path):
    module = _load_restore()
    base = {
        "source_chat_id": CHAT,
        "dest_chat_id": CHAT,
        "after": None,
        "before": None,
        "limit": None,
        "include_media": True,
    }
    same = dict(base)
    other = dict(base, before="2026-01-01")
    no_media = dict(base, include_media=False)
    assert module.job_id_for(base) == module.job_id_for(same)
    assert module.job_id_for(base) != module.job_id_for(other)
    assert module.job_id_for(base) != module.job_id_for(no_media)

    await _album(real_adapter, tmp_path / "media")
    run1, client1, _m, job_dir = await _run(real_adapter, tmp_path)
    assert run1.status == "complete"
    run2, client2, _m2, _jd = await _run(real_adapter, tmp_path, job_dir=job_dir, include_media=False)
    assert run2.status == "complete"
    assert len(glob.glob(os.path.join(job_dir, "restore-job-*.json"))) == 2


async def test_a_completed_job_is_a_noop_on_re_run(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")
    run1, _c1, _m, job_dir = await _run(real_adapter, tmp_path)
    assert run1.status == "complete"

    factory = AsyncMock(side_effect=AssertionError("nothing left to do"))
    run2, _c2, _m2, _jd = await _run(real_adapter, tmp_path, client_factory=factory, job_dir=job_dir)
    assert run2.status == "complete"
    factory.assert_not_awaited()


async def test_plan_drift_blocks_a_resume_when_the_backup_changes(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")
    send_file = AsyncMock(side_effect=[_msg(101), ConnectionError("gone")])
    run1, _c, _m, job_dir = await _run(
        real_adapter, tmp_path, send_file=send_file, send_message=_auto_send()
    )
    assert run1.status == "blocked"

    # A new archived message arrives before the job is finished: the bound plan
    # (order/text/media) no longer matches the database.
    await real_adapter.insert_message(
        {
            "id": 3,
            "chat_id": CHAT,
            "text": "Later note",
            "date": datetime(2026, 10, 5, 9, 0),
            "sender_name": "Fixture Sender",
        },
        account_id=1,
    )
    factory = AsyncMock(side_effect=AssertionError("drift must stop before connecting"))
    run2, _c2, _m2, _jd = await _run(real_adapter, tmp_path, client_factory=factory, job_dir=job_dir)
    assert run2.status == "active"
    assert run2.halt_reason == "plan_drift"
    factory.assert_not_awaited()


async def test_no_media_sends_only_text_and_binds_the_filter(real_adapter, tmp_path):
    await _album(real_adapter, tmp_path / "media")

    run1, client1, _m, job_dir = await _run(real_adapter, tmp_path, include_media=False)

    assert run1.status == "complete"
    client1.send_file.assert_not_awaited()
    assert _text_sends(client1) == [
        CAPTION,  # message 1, media skipped -> its header/body go as text
        TEXT_OF_2,
    ]
    data, _ = _job_on_disk(job_dir)
    assert data["params"]["include_media"] is False
    assert all(u["kind"] == "text" for item in data["items"] for u in item["units"])


def test_build_plan_items_keeps_media_order_text_unit_and_missing_warnings(tmp_path):
    module = _load_restore()
    media_base = tmp_path / "media"
    (media_base / str(CHAT)).mkdir(parents=True)
    (media_base / str(CHAT) / "a.jpg").write_bytes(b"a")
    (media_base / str(CHAT) / "c.jpg").write_bytes(b"c")

    def msg(message_id, text, rows):
        return {
            "id": message_id,
            "date": SENT.isoformat(),
            "sender": {"name": "S"},
            "text": text,
            "media": [{"media_id": mid} for mid, _p in rows],
            "media_files": [{"type": "photo", "path": p} for _mid, p in rows],
        }

    messages = [
        msg(1, "body", [("a", f"{CHAT}/a.jpg"), ("b", f"{CHAT}/b.jpg"), ("c", f"{CHAT}/c.jpg")]),
        msg(2, "all gone", [("d", f"{CHAT}/d.jpg")]),
        msg(3, "", []),
    ]
    items = module.build_plan_items(messages, True, str(media_base))

    assert [u["key"] for u in items[0]["units"]] == ["a", "c"]  # missing b keeps order
    assert [u["caption"] for u in items[0]["units"]] == [True, False]
    assert [f["key"] for f in items[0]["missing_files"]] == ["b"]
    # Every file missing: message still goes out as text.
    assert [u["kind"] for u in items[1]["units"]] == ["text"]
    assert [f["key"] for f in items[1]["missing_files"]] == ["d"]
    # No media and an empty body: the header still goes out, as in the old script.
    assert [u["kind"] for u in items[2]["units"]] == ["text"]
    assert items[2]["text"] == "[S - 2026-10-01 10:38]"
    assert len(items) == 3


@pytest.mark.parametrize(
    "outcome_kwargs,expected_rc",
    [
        ({"status": "complete"}, 0),
        ({"status": "dry_run"}, 0),
        ({"status": "aborted"}, 0),
        ({"status": "nothing"}, 0),
        ({"status": "active", "halt_reason": "error_threshold"}, 1),
        ({"status": "active", "halt_reason": "target_inaccessible"}, 1),
        ({"status": "active", "halt_reason": "interrupted"}, 130),
        ({"status": "blocked", "halt_reason": "unconfirmed_send"}, 2),
    ],
)
async def test_main_exit_codes(monkeypatch, outcome_kwargs, expected_rc):
    module = _load_restore()
    outcome = module.RestoreOutcome(**outcome_kwargs)
    monkeypatch.setattr(sys, "argv", ["restore_chat.py", "--chat", str(CHAT)])
    with patch.object(module, "restore_chat", AsyncMock(return_value=outcome)):
        rc = await module.main()
    assert rc == expected_rc
