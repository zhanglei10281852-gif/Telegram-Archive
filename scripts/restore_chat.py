#!/usr/bin/env python3
"""
Restore chat history from backup to Telegram as a resumable job.

⚠️  USE WITH CAUTION - This will send potentially thousands of messages!

IMPORTANT LIMITATIONS:
- Messages will be sent as YOU (the logged-in user), not the original sender
- Original timestamps are shown in message text, not as actual message time
- Media is re-uploaded as new files
- Telegram rate limits apply (~30 messages/minute for safety)

This is a Telegram API limitation - there is no way to send messages
as another user or with custom timestamps.

Resumable jobs
--------------
Every real run is bound to a job state file under
``$BACKUP_PATH/restore_jobs/`` (override with ``--job-dir``). The file binds
the source chat, destination chat, filters, the exact media order and the
confirmed result of every send (caption/text unit and each file unit, with
the destination message id Telegram returned).

Re-running with the same arguments resumes that job: already confirmed text,
captions and files are never sent again; only unconfirmed units go out, so a
run interrupted by a network drop or a killed process converges to the same
result as one successful run.

A send whose delivery cannot be determined (a connection/timeout error, a
process killed while the call was in flight, or a reply without a message id)
is recorded as *ambiguous*. The job stops there in a diagnosable state — it
never blindly resends (a duplicate would be certain if the call had landed)
and never skips ahead. Check the destination chat, then resume with either
``--resend-unconfirmed`` (the send did not arrive) or
``--mark-unconfirmed-sent`` (it did arrive).

``--dry-run`` builds the very same send plan but writes no job state and does
not connect to Telegram.

Usage:
    # Restore to the same chat (most common use case)
    python scripts/restore_chat.py --chat -1001234567890

    # Restore to a different destination
    python scripts/restore_chat.py --source-chat -1001234567890 --dest-chat -1009876543210

    # Dry run (show what would be sent without actually sending or writing a job)
    python scripts/restore_chat.py --chat -1001234567890 --dry-run

    # Restore only messages after a certain date
    python scripts/restore_chat.py --chat -1001234567890 --after 2024-01-01

    # Resume an interrupted job (same arguments; add one adjudication flag
    # only when the job is stopped at an unconfirmed send)
    python scripts/restore_chat.py --chat -1001234567890 --resend-unconfirmed

Environment variables (same as backup container):
    DB_TYPE, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
    or DB_PATH for SQLite

    TELEGRAM_API_ID, TELEGRAM_API_HASH, SESSION_NAME (or SESSION_PATH)
    BACKUP_PATH (for media files and restore_jobs/, default: /data/backups)
    RESTORE_JOB_DIR (override the job state directory)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError, SlowModeWaitError

from telegram_archive.config import build_telegram_client_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Telethon logs the connect address at INFO; with MTProxy that is the proxy host.
logging.getLogger("telethon").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Waits one send may sit out before the message counts as an error.
SEND_WAIT_RETRIES = 3

# A run stops after this many messages with a definite server-side failure.
# The loop counts one error per message and stops as the 21st is recorded,
# as the original one-shot script did.
MAX_MESSAGE_ERRORS = 20

JOB_STATE_VERSION = 1

# Unit states
PENDING = "pending"        # never sent, or the server definitively refused it
SENDING = "sending"        # call in flight: not confirmed yet (crash window)
SENT = "sent"              # Telegram answered with a destination message id
AMBIGUOUS = "ambiguous"    # call ended without a definite answer

# Job statuses
ACTIVE = "active"
BLOCKED = "blocked"
COMPLETE = "complete"

TERMINAL_STATES = (SENT,)


@dataclass
class RestoreOutcome:
    """How a run ended; main() turns it into an exit code."""

    status: str
    job_path: str | None = None
    sent_messages: int = 0
    sent_files: int = 0
    errors: int = 0
    pending: int = 0
    halt_reason: str | None = None


async def get_db_adapter():
    """Initialize and return database adapter."""
    from telegram_archive.db import DatabaseAdapter, init_database

    db_manager = await init_database()
    return DatabaseAdapter(db_manager)


async def get_telegram_client():
    """Initialize and return Telegram client."""
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id or not api_hash:
        raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables required")

    session_path = os.getenv("SESSION_PATH")
    if not session_path:
        session_name = os.getenv("SESSION_NAME", "telegram_backup")
        session_dir = os.getenv("SESSION_DIR", "/data/session")
        session_path = os.path.join(session_dir, session_name)

    client = TelegramClient(session_path, int(api_id), api_hash, **build_telegram_client_kwargs())
    await client.connect()

    if not await client.is_user_authorized():
        raise RuntimeError("Telegram session not authorized. Run the main backup first to authenticate.")

    return client


def format_message_header(sender_name: str, date: datetime | None) -> str:
    """Format the message header with sender and timestamp."""
    date_str = date.strftime("%Y-%m-%d %H:%M") if date else "Unknown date"
    return f"[{sender_name} - {date_str}]"


def parse_msg_date(msg: dict[str, Any]) -> datetime | None:
    """Parse message date from various formats."""
    d = msg.get("date")
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def render_full_text(msg: dict[str, Any]) -> str:
    """The header plus body the restore sends as text or caption."""
    sender_name = msg.get("sender", {}).get("name", "Unknown")
    header = format_message_header(sender_name, parse_msg_date(msg))
    text = msg.get("text", "") or ""
    return f"{header}\n{text}" if text else header


def caption_for(full_text: str) -> str:
    """Caption text for the first file of a media message (1024 char limit)."""
    return full_text[:1024] if len(full_text) <= 1024 else full_text[:1021] + "..."


def media_rows_in_export_order(msg: dict[str, Any]) -> list[tuple[Any, str | None, str | None]]:
    """``(media_id, type, stored_path)`` for every media row of a message.

    The export lists media downloaded first, then lowest id. ``media`` and
    ``media_files`` are built from the same ordered rows, so they line up by
    index. Rows without a stored path have no file to upload.
    """
    media = msg.get("media") or []
    media_files = msg.get("media_files") or []
    rows = []
    for meta, file_info in zip(media, media_files, strict=True):
        path = file_info.get("path")
        if path:
            rows.append((meta.get("media_id"), file_info.get("type"), path))
    return rows


# ---------------------------------------------------------------------------
# Send plan (shared by --dry-run and real runs)
# ---------------------------------------------------------------------------


async def load_filtered_messages(
    db,
    source_chat_id: int,
    *,
    after_date: datetime | None,
    before_date: datetime | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load a chat's export messages and apply the existing filters/sort."""
    messages = []
    async for msg in db.get_messages_for_export(source_chat_id, include_media=True):
        messages.append(msg)

    if after_date:
        messages = [m for m in messages if parse_msg_date(m) and parse_msg_date(m).replace(tzinfo=None) > after_date]

    if before_date:
        messages = [m for m in messages if parse_msg_date(m) and parse_msg_date(m).replace(tzinfo=None) < before_date]

    messages.sort(key=lambda m: parse_msg_date(m) or datetime.min)

    if limit and len(messages) > limit:
        messages = messages[:limit]
    return messages


def build_plan_items(
    messages: list[dict[str, Any]], include_media: bool, media_base_path: str
) -> list[dict[str, Any]]:
    """Turn filtered export messages into the ordered, bound send plan.

    One item per message that sends something. Its units preserve the media
    order of the files actually on disk and say which unit carries the text
    (the text message itself or the caption of the first file).

    A media row whose file is not on disk is recorded in ``missing_files`` and
    warned about, exactly as the one-shot script warned and skipped it; if no
    file of a message is on disk the message still goes out as text when it
    has any. A recovered file only enters a job built with ``--restart``.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        full_text = render_full_text(msg)
        units: list[dict[str, Any]] = []
        missing_files: list[dict[str, Any]] = []

        if include_media:
            for media_id, media_type, stored_path in media_rows_in_export_order(msg):
                abs_path = os.path.normpath(os.path.join(media_base_path, stored_path))
                if not os.path.exists(abs_path):
                    missing_files.append({"key": str(media_id) if media_id is not None else stored_path,
                                          "type": media_type, "path": stored_path})
                    continue
                units.append(
                    {
                        "key": str(media_id) if media_id is not None else stored_path,
                        "kind": "file",
                        "type": media_type,
                        "path": stored_path,
                        "caption": not units,  # the first file on disk carries the text
                        "state": PENDING,
                        "dest_message_id": None,
                        "dest_date": None,
                        "note": None,
                        "last_error": None,
                        "attempts": 0,
                    }
                )

        if not units and full_text.strip():
            # Text-only message (or a media message whose files are all gone)
            units.append(
                {
                    "key": "text",
                    "kind": "text",
                    "type": None,
                    "path": None,
                    "caption": False,
                    "state": PENDING,
                    "dest_message_id": None,
                    "dest_date": None,
                    "note": None,
                    "last_error": None,
                    "attempts": 0,
                }
            )

        if not units:
            # No files to send: the header always carries text, so this only
            # stays empty for pathological input; keep the old loop's skip.
            continue

        items.append(
            {
                "message_id": msg.get("id"),
                "sender": msg.get("sender", {}).get("name", "Unknown"),
                "date": msg.get("date"),
                "body": msg.get("text", "") or "",
                "text": full_text,
                "missing_files": missing_files,
                "units": units,
            }
        )
    return items


def plan_signature(items: list[dict[str, Any]]) -> list:
    """The parts of a plan that must not change between a job's runs.

    Message order, the rendered text/header of each message, and every send
    unit's identity, type, stored path and caption flag in media order. Media
    rows whose file was missing when the job was built are listed too, so a
    recovered file or a changed media set is detected as drift instead of
    silently changing what a resume sends.
    """
    return [
        (
            item["message_id"],
            item["text"],
            [("missing", f["key"], f["type"], f["path"], False) for f in item.get("missing_files", [])]
            + [(u["kind"], u["key"], u["type"], u["path"], u["caption"]) for u in item["units"]],
        )
        for item in items
    ]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def default_job_dir() -> str:
    override = os.getenv("RESTORE_JOB_DIR")
    if override:
        return override
    return os.path.join(os.getenv("BACKUP_PATH", "/data/backups"), "restore_jobs")


def job_id_for(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class PlanDriftError(RuntimeError):
    """The database no longer matches the plan a job was built from."""


class RestoreJob:
    """A restore run's durable state: filters, media order, send results."""

    def __init__(self, job_dir: str, data: dict[str, Any]):
        self.dir = job_dir
        self.data = data

    # ----- persistence -----------------------------------------------------

    @property
    def job_id(self) -> str:
        return self.data["job_id"]

    @property
    def path(self) -> str:
        return os.path.join(self.dir, f"restore-job-{self.job_id}.json")

    def save(self) -> None:
        """Atomically write the job state (temp file + replace + fsync)."""
        self.data["updated_at"] = _utc_now()
        os.makedirs(self.dir, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".restore-job-", suffix=".tmp", dir=self.dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def open(
        cls,
        job_dir: str,
        params: dict[str, Any],
        items: list[dict[str, Any]],
        *,
        restart: bool = False,
    ) -> "RestoreJob":
        """Load the job for ``params`` or create it from a fresh plan.

        On resume the fresh plan must still match the bound plan; otherwise
        the database changed under the job and resuming could skip or
        duplicate sends, so that is a hard, diagnosable error.
        """
        job_id = job_id_for(params)
        path = os.path.join(job_dir, f"restore-job-{job_id}.json")

        if not restart and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            job = cls(job_dir, data)
            if data.get("status") == COMPLETE:
                return job
            stored_sig = plan_signature(data["items"])
            fresh_sig = plan_signature(items)
            if stored_sig != fresh_sig:
                raise PlanDriftError(
                    f"The backup no longer matches the plan bound to job {job_id} "
                    f"({path}): message order, text or media changed since the job "
                    f"was created. Inspect the job file, or restart with --restart "
                    f"to build a new job (already confirmed sends are not undone)."
                )
            job._merge_fresh_items(items)
            job._adopt_interrupted_sends()
            # Durable even when this run stops at the blocked unit before
            # connecting: the next launch reads the same diagnosis.
            job.save()
            return job

        data = {
            "version": JOB_STATE_VERSION,
            "job_id": job_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "params": params,
            "status": ACTIVE,
            "halt_reason": None,
            "blocked": None,
            "errors": 0,
            "items": items,
        }
        job = cls(job_dir, data)
        job.save()
        return job

    def _merge_fresh_items(self, fresh_items: list[dict[str, Any]]) -> None:
        """Keep stored per-unit results, refresh display metadata from the DB."""
        stored = {item["message_id"]: item for item in self.data["items"]}
        for fresh in fresh_items:
            old = stored[fresh["message_id"]]
            old_units = {(u["kind"], u["key"]): u for u in old["units"]}
            merged = []
            for unit in fresh["units"]:
                previous = old_units[(unit["kind"], unit["key"])]
                unit["state"] = previous["state"]
                unit["dest_message_id"] = previous.get("dest_message_id")
                unit["dest_date"] = previous.get("dest_date")
                unit["note"] = previous.get("note")
                unit["last_error"] = previous.get("last_error")
                unit["attempts"] = previous.get("attempts", 0)
                merged.append(unit)
            fresh["units"] = merged
        self.data["items"] = fresh_items

    def _adopt_interrupted_sends(self) -> None:
        """A unit left ``sending`` never got its reply: treat it as ambiguous."""
        for item in self.data["items"]:
            for unit in item["units"]:
                if unit["state"] == SENDING:
                    unit["state"] = AMBIGUOUS
                    unit["last_error"] = "the process stopped while this send was in flight"
                    self.mark_blocked(item, unit, unit["last_error"])
                    return

    # ----- run state -------------------------------------------------------

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.data["items"]

    def mark_blocked(self, item: dict[str, Any], unit: dict[str, Any], error: str) -> None:
        self.data["status"] = BLOCKED
        self.data["halt_reason"] = "unconfirmed_send"
        self.data["blocked"] = {
            "message_id": item["message_id"],
            "unit_key": unit["key"],
            "kind": unit["kind"],
            "path": unit["path"],
            "caption": unit["caption"],
            "error": error,
            "at": _utc_now(),
        }

    def clear_blocked(self) -> None:
        self.data["status"] = ACTIVE
        self.data["halt_reason"] = None
        self.data["blocked"] = None

    def set_halt(self, reason: str) -> None:
        if self.data["status"] != BLOCKED:
            self.data["status"] = ACTIVE
        self.data["halt_reason"] = reason

    def counts(self) -> tuple[int, int, int]:
        """``(fully confirmed messages, confirmed file sends, pending units)``."""
        messages = 0
        files = 0
        pending = 0
        for item in self.data["items"]:
            states = [u["state"] for u in item["units"]]
            if all(state in TERMINAL_STATES for state in states):
                messages += 1
            files += sum(1 for u in item["units"] if u["state"] == SENT and u["kind"] == "file")
            pending += sum(1 for state in states if state != SENT and state != AMBIGUOUS)
        return messages, files, pending

    def ambiguous_units(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [
            (item, unit)
            for item in self.data["items"]
            for unit in item["units"]
            if unit["state"] == AMBIGUOUS
        ]

    def is_complete(self) -> bool:
        return all(
            u["state"] in TERMINAL_STATES for item in self.data["items"] for u in item["units"]
        )


async def send_with_wait_retry(send, *args, **kwargs):
    """Await a Telethon send, and after a flood or slow-mode wait send the same thing again.

    A message with several files is several sends; retrying the one that hit
    the wait keeps the files after it from being dropped. After
    ``SEND_WAIT_RETRIES`` waits the last wait error is raised.
    """
    for attempt in range(SEND_WAIT_RETRIES + 1):
        try:
            return await send(*args, **kwargs)
        except (FloodWaitError, SlowModeWaitError) as e:
            if attempt == SEND_WAIT_RETRIES:
                raise
            logger.warning(f"{type(e).__name__}: sleeping {e.seconds} seconds, then sending it again...")
            await asyncio.sleep(e.seconds + 1)


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def _media_base_path() -> str:
    # The export carries Media.file_path verbatim, and that column holds two
    # shapes: absolute (API sweep, realtime listener) and media-root-relative
    # (Telegram Desktop import). normpath keeps the joined result usable on
    # Windows; on POSIX it is a no-op.
    return os.path.normpath(os.path.join(os.getenv("BACKUP_PATH", "/data/backups"), "media"))


def _log_blocked_diagnostics(job: RestoreJob) -> dict[str, Any] | None:
    blocked = job.data.get("blocked")
    if not blocked:
        return None
    logger.error("=" * 60)
    logger.error("RESTORE STOPPED — ONE SEND IS UNCONFIRMED")
    logger.error("=" * 60)
    logger.error(f"Job:            {job.job_id}")
    logger.error(f"Job state file: {job.path}")
    logger.error(f"Source message: {blocked['message_id']}")
    logger.error(f"Unconfirmed:    {blocked['kind']} unit {blocked['unit_key']}")
    if blocked.get("path"):
        logger.error(f"File:           {blocked['path']}")
    logger.error(f"Carries text:   {'yes (caption)' if blocked.get('caption') else 'no'}")
    logger.error(f"Last error:     {blocked.get('error')}")
    logger.error(
        "Check whether this message/file actually arrived in the destination "
        "chat, then resume with:"
    )
    logger.error("  --resend-unconfirmed     if it did NOT arrive (it will be sent again)")
    logger.error("  --mark-unconfirmed-sent  if it DID arrive (record it without sending)")
    logger.error("=" * 60)
    return blocked


def _adjudicate_blocked_job(job: RestoreJob, *, resend: bool, mark_sent: bool) -> RestoreJob | None:
    """Resolve or surface the single ambiguous unit a stopped job sits on.

    Returns the job once it is safe to continue, or ``None`` when it must stay
    stopped. Flags are explicit operator decisions. On a terminal the operator
    is asked; with no terminal the job simply stays blocked.
    """
    ambiguous = job.ambiguous_units()
    if not ambiguous:
        job.clear_blocked()
        return job

    if resend and mark_sent:
        logger.error("Choose either --resend-unconfirmed or --mark-unconfirmed-sent, not both.")
        return None
    if resend:
        for item, unit in ambiguous:
            unit["state"] = PENDING
            unit["last_error"] = None
        job.clear_blocked()
        job.save()
        logger.info("Unconfirmed send will be retried as requested.")
        return job
    if mark_sent:
        for item, unit in ambiguous:
            unit["state"] = SENT
            unit["note"] = "operator confirmed delivery in the destination chat; no Telegram id"
        job.clear_blocked()
        job.save()
        logger.info("Unconfirmed send recorded as delivered as requested.")
        return job

    _log_blocked_diagnostics(job)
    if sys.stdin is not None and sys.stdin.isatty():
        answer = input(
            "Did the unconfirmed send arrive? Type RESEND (it did not), SENT (it did), "
            "or leave empty to stop: "
        ).strip().upper()
        if answer == "RESEND":
            return _adjudicate_blocked_job(job, resend=True, mark_sent=False)
        if answer == "SENT":
            return _adjudicate_blocked_job(job, resend=False, mark_sent=True)
    return None


async def restore_chat(
    source_chat_id: int,
    dest_chat_id: int,
    dry_run: bool = False,
    after_date: datetime | None = None,
    before_date: datetime | None = None,
    limit: int | None = None,
    delay: float = 2.0,
    include_media: bool = True,
    *,
    job_dir: str | None = None,
    restart: bool = False,
    resend_unconfirmed: bool = False,
    mark_unconfirmed_sent: bool = False,
) -> RestoreOutcome:
    """
    Restore messages from backup to Telegram chat, as a resumable job.

    Args:
        source_chat_id: Chat ID to read messages from (in backup DB)
        dest_chat_id: Chat ID to send messages to
        dry_run: If True, build and preview the plan without sending or writing state
        after_date: Only restore messages after this date
        before_date: Only restore messages before this date
        limit: Maximum number of messages to restore
        delay: Seconds to wait between messages (rate limiting)
        include_media: If True, also upload media files
        job_dir: Where job state files live (default: $BACKUP_PATH/restore_jobs)
        restart: Start a fresh job, ignoring any saved job for these parameters
        resend_unconfirmed: Resolve a blocked job by re-sending its ambiguous unit
        mark_unconfirmed_sent: Resolve a blocked job by accepting its ambiguous unit as delivered
    """
    logger.info("=" * 60)
    logger.info("TELEGRAM CHAT RESTORE")
    logger.info("=" * 60)
    logger.warning("⚠️  Messages will be sent as YOU, not the original sender!")
    logger.warning("⚠️  Original timestamps shown in text only.")
    logger.info(f"📎 Media: {'Included' if include_media else 'Skipped (text only)'}")
    if dry_run:
        logger.info("🔍 DRY RUN MODE - No messages will actually be sent")
    logger.info("=" * 60)

    # Initialize database
    logger.info("Connecting to database...")
    db = await get_db_adapter()

    # NOTE (#274): the chat ids logged below are deliberately kept. This is a
    # manually-run, one-shot, DESTRUCTIVE tool (it copies messages between
    # chats); the source and destination ids are the operator's own CLI
    # arguments, and echoing them — especially "about to send N to chat X" — is a
    # genuine safety confirmation before an irreversible action, not incidental
    # noise. Unlike the daemons this does not accumulate a shipped log stream.
    # This file is allow-listed in tests/test_no_account_pii_in_logs.py.
    # Get chat info
    chat = await db.get_chat_by_id(source_chat_id)
    if not chat:
        logger.error(f"Chat {source_chat_id} not found in backup database!")
        return RestoreOutcome(status="nothing")

    chat_name = chat.get("title") or chat.get("first_name") or f"Chat {source_chat_id}"
    logger.info(f"Source chat: {chat_name} (ID: {source_chat_id})")
    logger.info(f"Destination: {'Same chat' if source_chat_id == dest_chat_id else dest_chat_id}")

    # Get messages from backup (with media info) and apply existing filters
    logger.info("Loading messages from backup...")
    messages = await load_filtered_messages(
        db, source_chat_id, after_date=after_date, before_date=before_date, limit=limit
    )

    if not messages:
        logger.warning("No messages to restore after filtering!")
        return RestoreOutcome(status="nothing")

    logger.info(f"Found {len(messages)} messages to restore")

    media_base_path = _media_base_path()
    items = build_plan_items(messages, include_media, media_base_path)

    media_count = sum(1 for m in messages if include_media and media_rows_in_export_order(m))
    logger.info(f"\nWill restore {len(items)} messages ({media_count} with media)")
    logger.info(f"Estimated time: ~{len(items) * delay / 60:.1f} minutes (with {delay}s delay)")

    if dry_run:
        logger.info("\n--- DRY RUN PREVIEW (first 10 messages) ---")
        for item in items[:10]:
            header = format_message_header(item["sender"], parse_msg_date(item))
            text = item["body"][:80]
            media_info = ""
            if include_media:
                for missing in item.get("missing_files", []):
                    media_info += f" [📎 {missing['type'] or 'media'} ✗ MISSING]"
                for unit in item["units"]:
                    if unit["kind"] == "file":
                        media_info += f" [📎 {unit['type'] or 'media'} ✓]"
            logger.info(f"  {header}{media_info}\n    {text}")
        if len(items) > 10:
            logger.info(f"  ... and {len(items) - 10} more messages")
        logger.info("\nRun without --dry-run to actually send messages.")
        return RestoreOutcome(status="dry_run")

    # ----- durable job: bind source/dest/filters/media order/results --------
    job_dir = job_dir or default_job_dir()
    params = {
        "source_chat_id": source_chat_id,
        "dest_chat_id": dest_chat_id,
        "after": after_date.date().isoformat() if after_date else None,
        "before": before_date.date().isoformat() if before_date else None,
        "limit": limit,
        "include_media": include_media,
    }
    job_path = os.path.join(job_dir, f"restore-job-{job_id_for(params)}.json")
    restarting_with_history = restart and os.path.exists(job_path)
    try:
        job = RestoreJob.open(job_dir, params, items, restart=restart)
    except PlanDriftError as e:
        # describe_exception keeps the diagnostic text for our own error while
        # guaranteeing an OSError cannot smuggle a media path into the log.
        from telegram_archive.message_utils import describe_exception

        logger.error(describe_exception(e))
        return RestoreOutcome(status=ACTIVE, halt_reason="plan_drift", errors=0)

    already_messages, already_files, pending_units = job.counts()
    if job.status == COMPLETE:
        logger.info(f"Job {job.job_id} is already complete ({already_messages} messages, {already_files} files).")
        logger.info(f"Job state file: {job.path}")
        return RestoreOutcome(
            status=COMPLETE,
            job_path=job.path,
            sent_messages=already_messages,
            sent_files=already_files,
        )

    if restarting_with_history:
        logger.warning(
            "--restart starts a NEW job: messages already confirmed in Telegram "
            "are not undone and will be sent again."
        )

    logger.info(f"Resumable job: {job.job_id}")
    logger.info(f"Job state file: {job.path}")
    logger.info(
        f"Already confirmed: {already_messages} messages, {already_files} files; "
        f"{pending_units} units still to send."
    )

    for item in job.items:
        for missing in item.get("missing_files", []):
            logger.warning(
                f"Media file not found (message {item['message_id']}): {missing['path']}"
            )

    # A blocked job must be adjudicated before anything connects or sends.
    if job.status == BLOCKED:
        adjudicated = _adjudicate_blocked_job(
            job, resend=resend_unconfirmed, mark_sent=mark_unconfirmed_sent
        )
        if adjudicated is None:
            _messages, _files, pending = job.counts()
            return RestoreOutcome(
                status=BLOCKED,
                job_path=job.path,
                errors=job.data.get("errors", 0),
                pending=pending,
                halt_reason="unconfirmed_send",
            )
        _, _, pending_units = job.counts()

    # Confirm before proceeding
    logger.warning(f"\n⚠️  About to send {pending_units} unconfirmed units to chat {dest_chat_id}")
    logger.warning("⚠️  This action cannot be undone!")
    try:
        confirm = input("Type 'YES' to proceed: ")
    except EOFError:
        confirm = ""
    if confirm != "YES":
        logger.info("Aborted by user.")
        return RestoreOutcome(status="aborted", job_path=job.path)

    # The job exists on disk before any connection: an unreachable target or a
    # failed login still leaves a bound, resumable plan behind.
    job.save()

    client = None
    halt_reason = None
    interrupted = False
    try:
        # Initialize Telegram client
        logger.info("\nConnecting to Telegram...")
        try:
            client = await get_telegram_client()
        except Exception as e:
            halt_reason = "connect_failed"
            logger.error(f"Cannot connect to Telegram: {e}")
            job.set_halt(halt_reason)
            job.save()
            raise _StopRestore() from e
        logger.info("Logged in")

        # Verify destination chat exists
        try:
            dest_entity = await client.get_entity(dest_chat_id)
            dest_name = getattr(dest_entity, "title", None) or getattr(dest_entity, "first_name", "Unknown")
            logger.info(f"Destination verified: {dest_name}")
        except Exception as e:
            halt_reason = "target_inaccessible"
            logger.error(f"Cannot access destination chat {dest_chat_id}: {e}")
            job.set_halt(halt_reason)
            job.save()
            raise _StopRestore() from e

        halt_reason = await _run_sends(client, job, dest_chat_id, delay)
    except _StopRestore:
        pass  # halt_reason already recorded and job saved
    except KeyboardInterrupt:
        interrupted = True
        halt_reason = "interrupted"
        logger.warning("Interrupted — saving job progress before disconnecting...")
        try:
            job.set_halt(halt_reason)
            job.save()
        except Exception as save_error:  # never lose the disconnect
            logger.error(f"Could not save job state during interrupt: {save_error}")
    finally:
        if client is not None:
            await client.disconnect()
            logger.info("Telegram client disconnected")

    messages_done, files_done, pending_left = job.counts()
    ambiguous_left = len(job.ambiguous_units())

    if not interrupted and halt_reason is None and job.is_complete() and ambiguous_left == 0:
        job.data["status"] = COMPLETE
        job.data["halt_reason"] = None
        job.data["completed_at"] = _utc_now()
        job.save()
        logger.info("\n" + "=" * 60)
        logger.info("RESTORE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Messages sent: {messages_done}")
        logger.info(f"Media uploaded: {files_done}")
        logger.info(f"Errors: {job.data.get('errors', 0)}")
        logger.info(f"Job state file: {job.path}")
        logger.info("=" * 60)
        return RestoreOutcome(
            status=COMPLETE,
            job_path=job.path,
            sent_messages=messages_done,
            sent_files=files_done,
            errors=job.data.get("errors", 0),
        )

    # Not complete: say so plainly, with what remains and how to continue.
    if job.status == BLOCKED:
        _log_blocked_diagnostics(job)
        return RestoreOutcome(
            status=BLOCKED,
            job_path=job.path,
            sent_messages=messages_done,
            sent_files=files_done,
            errors=job.data.get("errors", 0),
            pending=pending_left,
            halt_reason="unconfirmed_send",
        )

    if halt_reason is None:
        halt_reason = "errors_remaining"
        job.set_halt(halt_reason)
        job.save()
    logger.info("\n" + "=" * 60)
    logger.info("RESTORE STOPPED (NOT COMPLETE)")
    logger.info("=" * 60)
    logger.info(f"Reason: {halt_reason}")
    logger.info(f"Messages confirmed: {messages_done}")
    logger.info(f"Files confirmed:    {files_done}")
    logger.info(f"Units unconfirmed:  {pending_left}")
    logger.info(f"Errors recorded:    {job.data.get('errors', 0)}")
    logger.info(f"Job state file:     {job.path}")
    logger.info("Re-run with the same arguments to continue; confirmed sends are not repeated.")
    logger.info("=" * 60)
    return RestoreOutcome(
        status=ACTIVE,
        job_path=job.path,
        sent_messages=messages_done,
        sent_files=files_done,
        errors=job.data.get("errors", 0),
        pending=pending_left,
        halt_reason=halt_reason,
    )


class _StopRestore(Exception):
    """Internal: progress is saved; jump to disconnect/summary."""


async def _dispatch_unit(client, dest_chat_id: int, item: dict[str, Any], unit: dict[str, Any]):
    """Perform the one Telegram call for a unit. Caller owns state transitions."""
    if unit["kind"] == "text":
        return await send_with_wait_retry(client.send_message, dest_chat_id, item["text"])

    abs_path = os.path.normpath(os.path.join(_media_base_path(), unit["path"]))
    if unit.get("caption"):
        return await send_with_wait_retry(
            client.send_file, dest_chat_id, abs_path, caption=caption_for(item["text"])
        )
    return await send_with_wait_retry(client.send_file, dest_chat_id, abs_path)


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


async def _run_sends(client, job: RestoreJob, dest_chat_id: int, delay: float) -> str | None:
    """Send every pending unit in plan order.

    Returns a halt reason, or ``None`` when the whole plan was walked. A unit
    the server definitively refuses stays ``pending`` and the run continues
    (one error per message, stop past ``MAX_MESSAGE_ERRORS``); a send with an
    unknown outcome blocks the whole job at that unit.
    """
    run_errors = 0
    over_threshold = False

    for item in job.items:
        if all(u["state"] in TERMINAL_STATES for u in item["units"]):
            continue

        files_sent_in_item = 0
        try:
            for unit in item["units"]:
                if unit["state"] in TERMINAL_STATES or unit["state"] == AMBIGUOUS:
                    if unit["kind"] == "file" and unit["state"] == SENT:
                        files_sent_in_item += 1
                    continue

                # Rate limiting between the files of one message (not before
                # its first send), exactly as the one-shot script paced it.
                if unit["kind"] == "file" and files_sent_in_item > 0:
                    await asyncio.sleep(delay)

                unit["attempts"] = unit.get("attempts", 0) + 1
                unit["state"] = SENDING
                unit["last_error"] = None
                job.save()

                try:
                    result = await _dispatch_unit(client, dest_chat_id, item, unit)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    # The SENDING marker is already on disk: the next run
                    # treats this unit as ambiguous instead of resending blind.
                    raise
                except (FloodWaitError, SlowModeWaitError) as e:
                    # Definitive: the wait was retried SEND_WAIT_RETRIES times
                    # and the server still says wait. Nothing was created.
                    unit["state"] = PENDING
                    unit["last_error"] = _error_text(e)
                    job.data["errors"] = job.data.get("errors", 0) + 1
                    run_errors += 1
                    job.save()
                    logger.warning(
                        f"{type(e).__name__} persisted for message {item['message_id']}: "
                        f"sleeping {e.seconds} seconds and leaving it unconfirmed"
                    )
                    await asyncio.sleep(e.seconds + 1)
                    raise _AbandonItem() from e
                except RPCError as e:
                    # A server response is a definite answer: rejected, not sent.
                    unit["state"] = PENDING
                    unit["last_error"] = _error_text(e)
                    job.data["errors"] = job.data.get("errors", 0) + 1
                    run_errors += 1
                    job.save()
                    logger.error(f"Telegram refused message {item['message_id']} ({unit['key']}): {e}")
                    raise _AbandonItem() from e
                except Exception as e:
                    # Connection drop, timeout, decode error, a reply without
                    # an id: delivery is unknown. Stop at this unit rather than
                    # risking a duplicate or skipping ahead.
                    unit["state"] = AMBIGUOUS
                    error = _error_text(e)
                    unit["last_error"] = error
                    job.mark_blocked(item, unit, error)
                    job.save()
                    _log_blocked_diagnostics(job)
                    return "unconfirmed_send"

                dest_id = getattr(result, "id", None)
                if dest_id is None:
                    error = "Telegram returned no message id for the send"
                    unit["state"] = AMBIGUOUS
                    unit["last_error"] = error
                    job.mark_blocked(item, unit, error)
                    job.save()
                    _log_blocked_diagnostics(job)
                    return "unconfirmed_send"

                unit["state"] = SENT
                unit["dest_message_id"] = dest_id
                dest_date = getattr(result, "date", None)
                unit["dest_date"] = dest_date.isoformat() if isinstance(dest_date, datetime) else None
                unit["last_error"] = None
                if unit["kind"] == "file":
                    files_sent_in_item += 1
                job.save()

            done_messages, done_files, pending = job.counts()
            if done_messages % 10 == 0 and done_messages:
                pct = done_messages / max(len(job.items), 1) * 100
                logger.info(
                    f"Progress: {done_messages}/{len(job.items)} messages ({pct:.1f}%) - {done_files} media files"
                )

            await asyncio.sleep(delay)

            # Definitive failures abandon the rest of this message but the run
            # walks the next one, as the one-shot script did.
        except _AbandonItem:
            if run_errors > MAX_MESSAGE_ERRORS:
                logger.error(f"More than {MAX_MESSAGE_ERRORS} message errors, stopping.")
                over_threshold = True
                break
            continue

    if over_threshold:
        job.set_halt("error_threshold")
        job.save()
        return "error_threshold"
    return None


class _AbandonItem(Exception):
    """Internal: this message had a definitive failure; try the next one."""


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d")


async def main():
    parser = argparse.ArgumentParser(
        description="Restore chat history from backup to Telegram as a resumable job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Restore to the same chat
    python scripts/restore_chat.py --chat -1001234567890

    # Dry run first (RECOMMENDED)
    python scripts/restore_chat.py --chat -1001234567890 --dry-run

    # Re-run after a network failure: same arguments, confirmed sends are skipped
    python scripts/restore_chat.py --chat -1001234567890

    # A job stopped at an unconfirmed send: check the destination chat first
    python scripts/restore_chat.py --chat -1001234567890 --resend-unconfirmed

    # Restore to different destination
    python scripts/restore_chat.py --source-chat -1001234567890 --dest-chat -1009876543210

    # Restore only recent messages
    python scripts/restore_chat.py --chat -1001234567890 --after 2024-01-01

    # Text only (no media)
    python scripts/restore_chat.py --chat -1001234567890 --no-media

⚠️  USE WITH CAUTION - Messages will be sent as YOU, not original senders!
        """,
    )

    # Chat selection (either --chat for same source/dest, or --source-chat/--dest-chat)
    chat_group = parser.add_mutually_exclusive_group(required=True)
    chat_group.add_argument("--chat", type=int, help="Chat ID to restore (sends back to same chat)")
    chat_group.add_argument(
        "--source-chat", type=int, help="Source chat ID (use with --dest-chat for different destination)"
    )

    parser.add_argument("--dest-chat", type=int, help="Destination chat ID (required if using --source-chat)")

    # Filters
    parser.add_argument("--after", type=str, help="Only restore messages after this date (YYYY-MM-DD)")
    parser.add_argument("--before", type=str, help="Only restore messages before this date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Maximum number of messages to restore")

    # Options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be sent without actually sending")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="Seconds between messages for rate limiting (default: 2.0)"
    )
    parser.add_argument("--no-media", action="store_true", help="Skip media files, restore text only")

    # Resumable job control
    parser.add_argument(
        "--job-dir",
        type=str,
        default=None,
        help="Directory for job state files (default: $RESTORE_JOB_DIR or $BACKUP_PATH/restore_jobs)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start a fresh job for these parameters instead of resuming (confirmed sends are not undone)",
    )
    adjudication = parser.add_mutually_exclusive_group()
    adjudication.add_argument(
        "--resend-unconfirmed",
        action="store_true",
        help="Resume a job stopped at an unconfirmed send by sending that unit again",
    )
    adjudication.add_argument(
        "--mark-unconfirmed-sent",
        action="store_true",
        help="Resume a job stopped at an unconfirmed send by recording it as delivered without sending",
    )

    args = parser.parse_args()

    # Determine source and destination
    if args.chat:
        source_chat_id = args.chat
        dest_chat_id = args.chat
    else:
        source_chat_id = args.source_chat
        if not args.dest_chat:
            parser.error("--dest-chat is required when using --source-chat")
        dest_chat_id = args.dest_chat

    # Parse dates
    after_date = parse_date(args.after) if args.after else None
    before_date = parse_date(args.before) if args.before else None

    outcome = await restore_chat(
        source_chat_id=source_chat_id,
        dest_chat_id=dest_chat_id,
        dry_run=args.dry_run,
        after_date=after_date,
        before_date=before_date,
        limit=args.limit,
        delay=args.delay,
        include_media=not args.no_media,
        job_dir=args.job_dir,
        restart=args.restart,
        resend_unconfirmed=args.resend_unconfirmed,
        mark_unconfirmed_sent=args.mark_unconfirmed_sent,
    )

    if outcome.status in (COMPLETE, "dry_run", "aborted", "nothing"):
        return 0
    if outcome.halt_reason == "interrupted":
        return 130
    if outcome.status == BLOCKED:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
