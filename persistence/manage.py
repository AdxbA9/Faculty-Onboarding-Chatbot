"""
Command line for the application database (post-freeze PF-13).

    python -m persistence.manage init                      create or upgrade the database (backup first when data is at risk)
    python -m persistence.manage check                     integrity, foreign keys, schema version, row counts
    python -m persistence.manage info                      where the database is and which backups exist
    python -m persistence.manage backup [DEST]             consistent copy through the online backup API (default: var/backups/)
    python -m persistence.manage restore SRC [--yes]       replace the database with a backup (stop the app first; the current
                                                           file is kept as .before-restore.<timestamp>.bak)
    python -m persistence.manage retention DAYS [--yes]    delete conversations not updated for DAYS days (all owners)

``--db PATH`` overrides ``APP_DB_PATH``. Nothing here prints user content or the cookie secret.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from typing import List, Optional

from . import db as _db


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _default_backup_dest(path: str) -> str:
    folder = os.path.join(os.path.dirname(path), "backups")
    return os.path.join(folder, "%s.%s%s" % (os.path.basename(path), time.strftime("%Y%m%d-%H%M%S"), _db.BACKUP_SUFFIX))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m persistence.manage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None, help="database file (default: APP_DB_PATH or var/app.db)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("check")
    sub.add_parser("info")
    p_backup = sub.add_parser("backup")
    p_backup.add_argument("dest", nargs="?", default=None)
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("src")
    p_restore.add_argument("--yes", action="store_true", help="confirm replacing the current database")
    p_ret = sub.add_parser("retention")
    p_ret.add_argument("days", type=int)
    p_ret.add_argument("--yes", action="store_true", help="confirm the deletion")
    args = parser.parse_args(argv)
    path = _db.db_path(args.db)

    try:
        if args.command == "init":
            _print(_db.initialize(path))
        elif args.command == "check":
            report = _db.check(path)
            _print(report)
            if report.get("exists") and (report.get("integrity") != "ok" or report.get("foreign_key_violations")):
                return 2
        elif args.command == "info":
            _print({"path": path, "exists": os.path.exists(path), "schema_version_supported": _db.SCHEMA_VERSION, "backups": _db.list_backups(path),
                    "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0})
        elif args.command == "backup":
            if not os.path.exists(path):
                print("no database at %s" % path, file=sys.stderr)
                return 1
            dest = args.dest or _default_backup_dest(path)
            _db.backup(path, dest)
            _print({"backup": os.path.abspath(dest), "size_bytes": os.path.getsize(dest)})
        elif args.command == "restore":
            if not args.yes:
                print("restore replaces %s with %s (the current file is kept as a .bak). Re-run with --yes to confirm." % (path, args.src), file=sys.stderr)
                return 1
            _print(_db.restore(args.src, path))
        elif args.command == "retention":
            from .store import Store
            if not args.yes:
                print("retention would delete every conversation not updated for %d days. Re-run with --yes to confirm." % args.days, file=sys.stderr)
                return 1
            store = Store(path)
            backup_path = _db.file_backup(path, "before-retention")
            removed = store.delete_older_than(args.days)
            _print({"removed_conversations": removed, "backup": backup_path})
    except (_db.DatabaseError, sqlite3.DatabaseError) as exc:
        print("database error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
