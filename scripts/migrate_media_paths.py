#!/usr/bin/env python3
"""
Migration Script: Normalize media folder paths to use marked IDs (negative for groups/channels)

This script migrates media folders, the database paths that point into them and
old-style avatar files from positive IDs ("35258041") to marked IDs
("-35258041" for basic groups, "-10035258041" for channels).

WHAT IT DOES
1. Builds ONE plan from the live database and media directory before touching
   anything: chat folders to relocate, every same-name target compared, the
   exact number of database rows to rewrite (media.file_path,
   media_versions.file_path when the table exists, and messages.media_path on
   pre-v6 archives), and avatar files to rename.
2. A same-name target is never overwritten and its source is never deleted
   blindly. The two entries are compared by content (SHA-256, following
   symlinks):
     - identical bytes  -> the source copy is redundant and gets removed;
     - different content, mismatched types or an unreadable entry -> the run
       STOPS before any destructive operation and reports the conflict.
3. Only a plan with no conflicts is applied. Each chat is one recoverable
   boundary: its filesystem changes land first (atomic renames; duplicates are
   removed only after the hash check), then its database rows are rewritten and
   committed in one transaction. Avatars are filesystem-only and are applied
   last.

CRASH RECOVERY
The plan is rebuilt from the actual disk and database state on every run, so a
process killed between a rename and its database commit cannot strand an
archive: files already at the target are simply found there, files still in the
old folder are relocated, and the idempotent path rewrite is committed when the
filesystem side is complete. A rerun after any interruption finishes in exactly
the state a single successful run would have produced. Nothing is reported as
complete while any stage is still pending; rerunning is the documented resume.

USAGE
    # Dry run (builds the same plan, writes nothing):
    python scripts/migrate_media_paths.py --dry-run

    # Actually migrate:
    python scripts/migrate_media_paths.py

    # With custom paths:
    python scripts/migrate_media_paths.py --media-path /path/to/media --db-url postgresql://...

Exit codes: 0 success (including nothing-to-do and dry-run without conflicts),
1 a stage failed (rerun to resume), 2 the plan has conflicts (nothing written).

BACKUP FIRST!
    - Backup your media folder
    - Backup your database
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass, field

# Import the installed package the same way the other maintenance scripts do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from telegram_archive.web.media_utils import derive_stale_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Plan actions.
MOVE = "move"  # rename source into the (absent) target
DEDUP = "dedup"  # target already holds byte-identical content; remove source
CONFLICT = "conflict"  # cannot prove equivalence; blocks the whole run
MERGE_DIR = "merge-dir"  # both sides are directories; children are merged

# Conflict reasons.
DIFFERENT_CONTENT = "different-content"
TYPE_MISMATCH = "type-mismatch"
UNREADABLE = "unreadable"

MEDIA_SCOPE = "media"
AVATAR_SCOPE = "avatar"

_HASH_CHUNK = 65536

# Path-bearing columns, keyed by table. Each tuple is (table, column). The
# identifiers are constants, never user input, so interpolating them into SQL is
# safe; every compared/rewritten VALUE is a bound parameter. ``messages`` is
# pre-v6 (the column was normalized into media and dropped in schema v6);
# ``media_versions`` joined in 036. Tables or columns absent from the connected
# database are skipped.
PATH_COLUMNS = (
    ("media", "file_path"),
    ("media_versions", "file_path"),
    ("messages", "media_path"),
)

AVATAR_NAME_RE = re.compile(r"^(\d+)_(\d+)\.jpg$", re.ASCII)


class MigrationError(Exception):
    """A filesystem or database stage failed; the run must not report success."""


# ---------------------------------------------------------------------------
# Plan data
# ---------------------------------------------------------------------------


@dataclass
class Relocation:
    """One planned filesystem change (or a blocking conflict)."""

    scope: str
    source: str
    target: str
    action: str
    # Display name. Avatar filenames embed the chat id, so this stays generic
    # for the avatar scope (#274: chat ids never reach the logs).
    label: str
    reason: str | None = None


@dataclass
class ChatPlan:
    old_folder: str
    new_folder: str
    old_dir: str | None
    new_dir_present: bool
    relocations: list[Relocation] = field(default_factory=list)
    # Rows to rewrite per table, keyed "table.column". Zero for tables that are
    # absent from this database or already canonical.
    db_rows: dict[str, int] = field(default_factory=dict)

    @property
    def rename_whole(self) -> bool:
        """Old folder present, new folder absent: one atomic directory rename."""
        return self.old_dir is not None and not self.new_dir_present

    @property
    def filesystem_pending(self) -> bool:
        return self.old_dir is not None

    @property
    def moves(self) -> list[Relocation]:
        return [r for r in self.relocations if r.action == MOVE]

    @property
    def dedups(self) -> list[Relocation]:
        return [r for r in self.relocations if r.action == DEDUP]


@dataclass
class MigrationPlan:
    chats: list[ChatPlan] = field(default_factory=list)
    avatars: list[Relocation] = field(default_factory=list)
    # Planning-level problems that forbid execution (e.g. two chats whose
    # legacy folder name collides). PII-safe text only.
    blocked: list[str] = field(default_factory=list)
    available_columns: set[str] = field(default_factory=set)

    @property
    def conflicts(self) -> list[Relocation]:
        found = [r for chat in self.chats for r in chat.relocations if r.action == CONFLICT]
        found.extend(r for r in self.avatars if r.action == CONFLICT)
        return found

    @property
    def executable(self) -> bool:
        return not self.blocked and not self.conflicts

    @property
    def total_db_rows(self) -> int:
        return sum(sum(chat.db_rows.values()) for chat in self.chats)


# ---------------------------------------------------------------------------
# Content comparison
# ---------------------------------------------------------------------------


def hash_file(path: str) -> str | None:
    """SHA-256 hex digest of a file, following symlinks; None if unreadable."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_HASH_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _equivalent_entries(source: str, target: str) -> tuple[bool, str | None]:
    """Whether an existing same-name target is safe to treat as a duplicate.

    Returns (equivalent, conflict_reason). Both paths lexist (the target name
    is taken); the job is to prove they hold the same thing, never to guess.
    """
    source_link = os.path.islink(source)
    target_link = os.path.islink(target)

    # Two symlinks: identical link text means the identical directory entry,
    # including when both are dangling. Diverging text (relative vs absolute,
    # different targets) cannot be proven equal for links to directories, so
    # fall through to content comparison where possible.
    if source_link and target_link:
        try:
            if os.readlink(source) == os.readlink(target):
                return True, None
        except OSError:
            return False, UNREADABLE

    source_is_dir = os.path.isdir(source)  # follows links
    target_is_dir = os.path.isdir(target)
    if source_is_dir or target_is_dir:
        # Both directories are handled by the tree walk; a dir vs non-dir is a
        # hard type mismatch.
        if source_is_dir and target_is_dir:
            return True, None
        return False, TYPE_MISMATCH

    if not os.path.isfile(source) or not os.path.isfile(target):
        return False, TYPE_MISMATCH

    source_hash = hash_file(source)
    target_hash = hash_file(target)
    if source_hash is None or target_hash is None:
        return False, UNREADABLE
    if source_hash != target_hash:
        return False, DIFFERENT_CONTENT
    return True, None


def classify_entry(scope: str, source: str, target: str, label: str) -> Relocation:
    """Plan one old-folder entry against its same-name target."""
    if not os.path.lexists(target):
        return Relocation(scope, source, target, MOVE, label)
    equivalent, reason = _equivalent_entries(source, target)
    if equivalent:
        return Relocation(scope, source, target, DEDUP, label)
    return Relocation(scope, source, target, CONFLICT, label, reason)


# ---------------------------------------------------------------------------
# Filesystem planning
# ---------------------------------------------------------------------------


def plan_directory_merge(old_dir: str, new_dir: str) -> list[Relocation]:
    """Compare two existing chat folders and plan their merge.

    A subdirectory missing on the target side is moved as one subtree; two
    directories are descended into. Every other same-name entry goes through
    content comparison.
    """
    planned: list[Relocation] = []

    def walk(current_old: str, current_new: str, prefix: str) -> None:
        try:
            entries = sorted(os.scandir(current_old), key=lambda entry: entry.name)
        except OSError as exc:
            raise MigrationError(f"cannot read a media folder: {type(exc).__name__}") from exc
        for entry in entries:
            source = entry.path
            target = os.path.join(current_new, entry.name)
            label = f"{prefix}{entry.name}"
            try:
                source_is_dir = entry.is_dir(follow_symlinks=False)
                target_exists = os.path.lexists(target)
                target_is_real_dir = target_exists and os.path.isdir(target) and not os.path.islink(target)
            except OSError as exc:
                raise MigrationError(f"cannot stat a media folder entry: {type(exc).__name__}") from exc

            if source_is_dir:
                if not target_exists:
                    # Whole subtree relocates in one atomic rename; do not
                    # descend (nothing on the target side to compare against).
                    planned.append(Relocation(MEDIA_SCOPE, source, target, MOVE, label))
                elif target_is_real_dir:
                    walk(source, target, f"{label}/")
                else:
                    # Target name taken by a file/symlink.
                    planned.append(Relocation(MEDIA_SCOPE, source, target, CONFLICT, label, TYPE_MISMATCH))
            else:
                planned.append(classify_entry(MEDIA_SCOPE, source, target, label))

    walk(old_dir, new_dir, "")
    return planned


def plan_avatars(media_path: str) -> list[Relocation]:
    """Plan renames of positive-id chat avatars to their marked form."""
    avatar_dir = os.path.join(media_path, "avatars", "chats")
    if not os.path.isdir(avatar_dir):
        return []

    planned: list[Relocation] = []
    try:
        names = sorted(os.listdir(avatar_dir))
    except OSError as exc:
        raise MigrationError(f"cannot read the avatar folder: {type(exc).__name__}") from exc

    for name in names:
        match = AVATAR_NAME_RE.match(name)
        if not match:
            continue  # already marked or a different avatar naming scheme
        target_name = f"-{match.group(1)}_{match.group(2)}.jpg"
        source = os.path.join(avatar_dir, name)
        target = os.path.join(avatar_dir, target_name)
        planned.append(classify_entry(AVATAR_SCOPE, source, target, "an avatar"))
    return planned


# ---------------------------------------------------------------------------
# Database: dialect-portable path matching
# ---------------------------------------------------------------------------
#
# Stored paths have two shapes (web/media_utils.resolve_stored_media_path):
# absolute from the capture layer (".../media/<folder>/<file>") and
# media-root-relative from the importer ("<folder>/<file>"). The old script
# matched only the absolute "/media/<folder>/" shape, so importer rows were
# never rewritten.
#
# The folder is matched as a COMPLETE path segment:
#   "<old>/%"  relative prefix, or "%/<old>/%" a segment inside an absolute
# path. Folder names are digits only, so LIKE has no escaping concerns.
# Both SQLite and PostgreSQL provide substr() (1-based), || and replace().


def _path_sql_params(old_folder: str, new_folder: str) -> dict[str, str | int]:
    return {
        "rel_like": f"{old_folder}/%",
        "seg_like": f"%/{old_folder}/%",
        "seg": f"/{old_folder}/",
        "new_seg": f"/{new_folder}/",
        "new_prefix": f"{new_folder}/",
        "cut": len(old_folder) + 2,  # 1-based start of the filename in "<old>/<file>"
    }


def _where_clause(column: str) -> str:
    return f"{column} IS NOT NULL AND ({column} LIKE :rel_like OR {column} LIKE :seg_like)"


def count_paths_sql(table: str, column: str) -> str:
    return f"SELECT COUNT(*) FROM {table} WHERE chat_id = :cid AND {_where_clause(column)}"


def update_paths_sql(table: str, column: str) -> str:
    return f"""
        UPDATE {table}
        SET {column} = CASE
            WHEN {column} LIKE :rel_like THEN :new_prefix || substr({column}, :cut)
            ELSE replace({column}, :seg, :new_seg)
        END
        WHERE chat_id = :cid AND {_where_clause(column)}
    """


async def _table_columns(conn, table: str) -> set[str]:
    def _inspect(sync_conn) -> set[str]:
        return {column["name"] for column in sa.inspect(sync_conn).get_columns(table)}

    try:
        return await conn.run_sync(_inspect)
    except sa.exc.NoSuchTableError:
        return set()


async def _negative_chats(conn) -> list[int]:
    result = await conn.execute(text("SELECT id FROM chats WHERE id < 0 ORDER BY id"))
    return [row[0] for row in result.fetchall()]


# ---------------------------------------------------------------------------
# Plan construction (read-only)
# ---------------------------------------------------------------------------


async def build_plan(conn, media_path: str) -> MigrationPlan:
    """Inspect the database and media folder and produce the one migration plan."""
    plan = MigrationPlan()

    available: set[str] = set()
    for table, column in PATH_COLUMNS:
        if column in await _table_columns(conn, table):
            available.add(f"{table}.{column}")
    plan.available_columns = available

    chat_ids = await _negative_chats(conn)

    # Two marked chats can share one legacy positive folder name (a basic group
    # -X and a channel -(10^12+X) both descended from "X"). Their database rows
    # are rewritten independently and safely, but if the positive folder still
    # exists its files cannot be split between the two targets from names
    # alone. Detect that here and make it a planning error.
    claimants: dict[str, list[int]] = {}
    for chat_id in chat_ids:
        old_folder = derive_stale_folder(chat_id)
        if old_folder:
            claimants.setdefault(old_folder, []).append(chat_id)

    ambiguous = {folder for folder, ids in claimants.items() if len(ids) > 1}

    for chat_id in chat_ids:
        old_folder = derive_stale_folder(chat_id)
        if old_folder is None:
            continue
        new_folder = str(chat_id)
        old_dir_path = os.path.join(media_path, old_folder)
        new_dir_path = os.path.join(media_path, new_folder)
        old_dir = old_dir_path if os.path.isdir(old_dir_path) else None
        new_dir_present = os.path.isdir(new_dir_path)

        chat_plan = ChatPlan(
            old_folder=old_folder, new_folder=new_folder, old_dir=old_dir, new_dir_present=new_dir_present
        )

        for table, column in PATH_COLUMNS:
            key = f"{table}.{column}"
            if key not in available:
                continue
            result = await conn.execute(
                text(count_paths_sql(table, column)),
                {"cid": chat_id, **_path_sql_params(old_folder, new_folder)},
            )
            chat_plan.db_rows[key] = int(result.scalar() or 0)

        if old_dir is not None:
            if old_folder in ambiguous:
                plan.blocked.append(
                    "a legacy media folder is claimed by two chats with different marked ids; "
                    "split or resolve it manually, then rerun"
                )
            elif new_dir_present:
                chat_plan.relocations = plan_directory_merge(old_dir, new_dir_path)
            # else: whole rename; relocations stay empty

        if chat_plan.filesystem_pending or chat_plan.db_rows:
            plan.chats.append(chat_plan)

    # One blocked message per contested folder, not per claimant.
    plan.blocked = list(dict.fromkeys(plan.blocked))
    plan.avatars = plan_avatars(media_path)
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _rename(source: str, target: str) -> None:
    try:
        os.rename(source, target)
    except OSError as exc:
        raise MigrationError(f"filesystem rename failed: {type(exc).__name__}") from exc


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError as exc:
        raise MigrationError(f"filesystem remove failed: {type(exc).__name__}") from exc


def _remove_empty_dirs(root: str) -> int:
    """Remove now-empty directory trees bottom-up, including root. Best effort:
    a non-empty directory is left for the next run to converge."""
    removed = 0
    for current, subdirs, files in os.walk(root, topdown=False):
        try:
            os.rmdir(current)
            removed += 1
        except OSError:
            pass
    return removed


def apply_chat_filesystem(chat_plan: ChatPlan) -> dict[str, int]:
    """Relocate one chat's files. Renames are atomic; removals only happen for
    entries the plan proved byte-identical to their target."""
    stats = {"moves": 0, "dedups": 0, "empty_dirs_removed": 0}
    if chat_plan.old_dir is None:
        return stats

    if chat_plan.rename_whole:
        _rename(chat_plan.old_dir, os.path.join(os.path.dirname(chat_plan.old_dir), chat_plan.new_folder))
        stats["moves"] += 1
        return stats

    for relocation in chat_plan.relocations:
        if relocation.action == MOVE:
            _rename(relocation.source, relocation.target)
            stats["moves"] += 1
        elif relocation.action == DEDUP:
            if os.path.isdir(relocation.source) and not os.path.islink(relocation.source):
                # Emptied by the child relocations; the sweep below removes it.
                continue
            _remove_file(relocation.source)
            stats["dedups"] += 1
        # MERGE_DIR and CONFLICT need no action (conflicts stop the run earlier).

    stats["empty_dirs_removed"] = _remove_empty_dirs(chat_plan.old_dir)
    return stats


async def rewrite_chat_paths(conn, chat_plan: ChatPlan) -> int:
    """Rewrite every still-legacy path row for one chat inside one transaction."""
    updated = 0
    for table, column in PATH_COLUMNS:
        key = f"{table}.{column}"
        if not chat_plan.db_rows.get(key):
            continue
        result = await conn.execute(
            text(update_paths_sql(table, column)),
            {"cid": int(chat_plan.new_folder), **_path_sql_params(chat_plan.old_folder, chat_plan.new_folder)},
        )
        updated += result.rowcount or chat_plan.db_rows[key]
    return updated


def apply_avatars(relocations: list[Relocation]) -> dict[str, int]:
    stats = {"moves": 0, "dedups": 0}
    for relocation in relocations:
        if relocation.action == MOVE:
            _rename(relocation.source, relocation.target)
            stats["moves"] += 1
        elif relocation.action == DEDUP:
            _remove_file(relocation.source)
            stats["dedups"] += 1
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_plan(plan: MigrationPlan, dry_run: bool) -> None:
    whole_renames = sum(1 for c in plan.chats if c.rename_whole)
    entry_moves = sum(len(c.moves) for c in plan.chats)
    dedups = sum(len(c.dedups) for c in plan.chats)
    avatar_moves = sum(1 for r in plan.avatars if r.action == MOVE)
    avatar_dedups = sum(1 for r in plan.avatars if r.action == DEDUP)

    logger.info("-" * 70)
    logger.info("MIGRATION PLAN")
    logger.info(f"  Chat folders involved:         {len(plan.chats)}")
    logger.info(f"    whole-folder renames:        {whole_renames}")
    logger.info(f"    files/subtrees to move:      {entry_moves}")
    logger.info(f"    identical copies to remove:  {dedups}")
    per_table = {}
    for chat in plan.chats:
        for key, count in chat.db_rows.items():
            per_table[key] = per_table.get(key, 0) + count
    for key in sorted(per_table):
        logger.info(f"    DB rows in {key:<26} {per_table[key]}")
    logger.info(f"  Database rows total:           {plan.total_db_rows}")
    logger.info(f"  Avatars to rename:             {avatar_moves}")
    logger.info(f"  Avatar duplicates to remove:   {avatar_dedups}")
    if dry_run:
        logger.info("  (dry run: this plan is not applied)")
    logger.info("-" * 70)


def report_conflicts(plan: MigrationPlan) -> None:
    conflicts = plan.conflicts
    logger.error("=" * 70)
    logger.error(f"STOPPED: {len(conflicts)} same-name conflict(s) and {len(plan.blocked)} planning problem(s)")
    logger.error("Nothing was moved, removed or updated. Resolve these and rerun.")
    for message in plan.blocked:
        logger.error(f"  planning: {message}")
    for relocation in conflicts:
        if relocation.scope == AVATAR_SCOPE:
            # The filename embeds the chat id; report the kind only.
            logger.error(f"  avatar target: {relocation.reason}")
        else:
            logger.error(f"  media file {relocation.label}: {relocation.reason}")
    logger.error("=" * 70)


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    status: str  # "applied" | "dry-run" | "conflicts" | "failed"
    stats: dict = field(default_factory=dict)
    error: str | None = None


async def migrate(db_url: str, media_path: str, dry_run: bool = True) -> RunResult:
    """Build the plan and, unless this is a dry run or conflicts block it, apply it."""

    logger.info("=" * 70)
    logger.info("Media Path Migration Script - Normalize to Marked IDs")
    logger.info("=" * 70)
    if dry_run:
        logger.info("DRY RUN MODE - the same plan is built, but nothing is written")
    else:
        logger.warning("LIVE MODE - changes will be applied")
    logger.info(f"Media path: {media_path}")

    engine = create_async_engine(db_url, echo=False)
    stats = {
        "folders_renamed": 0,
        "entries_moved": 0,
        "duplicates_removed": 0,
        "db_rows_updated": 0,
        "avatars_moved": 0,
        "avatar_duplicates_removed": 0,
    }

    try:
        async with engine.connect() as conn:
            plan = await build_plan(conn, media_path)

        log_plan(plan, dry_run)

        if not plan.executable:
            report_conflicts(plan)
            return RunResult("conflicts", stats)

        if dry_run:
            logger.info("DRY RUN complete - no changes were made")
            return RunResult("dry-run", stats)

        # Per-chat recoverable boundary: filesystem first, database commit
        # second. A crash in between is repaired by the next run, which finds
        # the files already at the target and only commits the rewrite.
        for index, chat_plan in enumerate(plan.chats, start=1):
            if chat_plan.filesystem_pending:
                if chat_plan.rename_whole:
                    logger.info(f"[{index}/{len(plan.chats)}] Renaming one media folder to marked format")
                else:
                    logger.info(f"[{index}/{len(plan.chats)}] Merging one media folder into its marked folder")
                fs_stats = apply_chat_filesystem(chat_plan)
                stats["folders_renamed"] += 1
                stats["entries_moved"] += fs_stats["moves"]
                stats["duplicates_removed"] += fs_stats["dedups"]

            if chat_plan.db_rows:
                async with engine.begin() as conn:
                    stats["db_rows_updated"] += await rewrite_chat_paths(conn, chat_plan)
                logger.info(f"[{index}/{len(plan.chats)}] Database paths committed")

        if plan.avatars:
            logger.info("Migrating avatars...")
            avatar_stats = apply_avatars(plan.avatars)
            stats["avatars_moved"] += avatar_stats["moves"]
            stats["avatar_duplicates_removed"] += avatar_stats["dedups"]

    except (MigrationError, sa.exc.SQLAlchemyError) as exc:
        # Deliberately type-name safe messages already; never interpolate raw
        # OSError text because it carries a media path with a chat-id folder.
        error = str(exc) if isinstance(exc, MigrationError) else f"database transaction failed: {type(exc).__name__}"
        logger.error("")
        logger.error(f"MIGRATION INTERRUPTED: {error}")
        logger.error(
            "The archive is left in a resumable state. Fix the cause and rerun; "
            "the next run continues from the current state."
        )
        return RunResult("failed", stats, error)
    finally:
        await engine.dispose()

    logger.info("")
    logger.info("=" * 70)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Folders renamed:          {stats['folders_renamed']}")
    logger.info(f"Files/subtrees moved:     {stats['entries_moved']}")
    logger.info(f"Duplicate copies removed: {stats['duplicates_removed']}")
    logger.info(f"DB paths updated:         {stats['db_rows_updated']}")
    logger.info(f"Avatars renamed:          {stats['avatars_moved']}")
    logger.info(f"Avatar duplicates removed:{stats['avatar_duplicates_removed']}")
    logger.info("")
    logger.info("Migration complete - rerunning is safe and performs no further work")
    return RunResult("applied", stats)


def get_database_url() -> str:
    """Build database URL from environment variables (same logic as Config)."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    db_type = os.environ.get("DB_TYPE", "sqlite").lower()

    if db_type == "postgresql":
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "telegram_backup")
        user = os.environ.get("POSTGRES_USER", "telegram")
        password = os.environ.get("POSTGRES_PASSWORD", "telegram")
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

    backup_path = os.environ.get("BACKUP_PATH", "/data/backups")
    return f"sqlite:///{backup_path}/telegram_backup.db"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate media paths, folders and avatars to marked ids, with conflict checking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the plan and preview changes without writing")
    parser.add_argument(
        "--media-path",
        default=os.environ.get("MEDIA_PATH", "/data/backups/media"),
        help="Path to media directory (default: $MEDIA_PATH or /data/backups/media)",
    )
    parser.add_argument(
        "--db-url", default=None, help="Database URL (default: built from env vars like DB_TYPE, POSTGRES_*)"
    )
    args = parser.parse_args()

    db_url = args.db_url if args.db_url else get_database_url()

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("sqlite://"):
        db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    result = asyncio.run(migrate(db_url, args.media_path, args.dry_run))
    if result.status == "conflicts":
        sys.exit(2)
    if result.status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
