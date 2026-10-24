"""Delete matching ``.log`` files on a remote server.

The counterpart of ``pull_logs.py``: instead of downloading, it *removes* the
``*.log`` files in a remote directory whose filename contains a given substring
(e.g. clean up a stale batch of ``search_cifar100_*.log`` runs on the server).

Deletion is irreversible, so the script is safe by default:

* it first lists every file that would be deleted, and
* unless ``--yes`` is given, it asks for confirmation before touching anything.

Use ``--dry-run`` to only list matches without deleting (or prompting).

Connection / auth details are shared with ``pull_logs.py`` in ``tool/_ssh.py``.

Examples
--------
    # List what would be deleted, without deleting
    python tool/delete_logs.py --server alice@server1 --remote-dir '~/llm-nas' \
        --contains cifar100 --dry-run

    # Delete after an interactive [y/N] confirmation
    python tool/delete_logs.py --server alice@server1 --remote-dir '~/llm-nas' \
        --contains cifar100

    python tool/delete_logs.py  --server jucilab@172.18.144.9 --remote-dir '~/zhengruitao/LLM-NAS' \
      --contains cifar100  --port 20300
    # Delete without prompting (e.g. from a script)
    python tool/delete_logs.py --server alice@server1 --remote-dir '~/llm-nas' \
        --contains cifar100 --yes
"""

from __future__ import annotations

import argparse
import os
import sys

from _ssh import (
    add_connection_args,
    connect,
    expand_remote_dir,
    hint_if_local_tilde,
    list_remote_logs,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="delete_logs.py",
        description="Delete remote *.log files whose name contains a substring.",
    )
    add_connection_args(p)
    p.add_argument(
        "--contains",
        required=True,
        metavar="SUBSTR",
        help="Only delete *.log files whose filename contains this substring",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="List matching files without deleting them"
    )
    p.add_argument(
        "--yes", "-y", action="store_true", help="Delete without the confirmation prompt"
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    return p.parse_args(argv)


def _confirm(files: list[str], server: str, remote_dir: str) -> bool:
    """Ask for confirmation, listing every file first; return True only on a clear yes."""
    print(f"\nAbout to delete {len(files)} file(s) from {server}:{remote_dir}:")
    for path in files:
        print(f"  {os.path.basename(path)}")
    try:
        answer = input("\nProceed? [y/N] ").strip().lower()
    except EOFError:
        # Non-interactive stdin: never delete without an explicit --yes.
        answer = "n"
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    client = connect(args)
    try:
        sftp = client.open_sftp()
        try:
            remote_dir = expand_remote_dir(client, args.remote_dir)
            try:
                all_logs = list_remote_logs(sftp, remote_dir)
            except OSError as exc:
                print(
                    f"error: could not list {args.server}:{args.remote_dir} — {exc}",
                    file=sys.stderr,
                )
                hint_if_local_tilde(args.remote_dir)
                return 1

            matching = [path for path in all_logs if args.contains in os.path.basename(path)]

            if not matching:
                print(
                    f"no *.log files under {args.server}:{args.remote_dir} contain "
                    f"{args.contains!r} ({len(all_logs)} *.log files scanned)"
                )
                return 1

            if args.dry_run:
                print(
                    f"{len(matching)} matching *.log file(s) on "
                    f"{args.server}:{args.remote_dir} (would be deleted):"
                )
                for path in matching:
                    print(f"  {os.path.basename(path)}")
                return 0

            if not args.yes and not _confirm(matching, args.server, args.remote_dir):
                print("Aborted — nothing deleted.")
                return 0

            deleted: list[str] = []
            failed: list[str] = []
            for path in matching:
                name = os.path.basename(path)
                try:
                    sftp.remove(path)
                    deleted.append(name)
                    if not args.quiet:
                        print(f"deleted {name}")
                except OSError as exc:
                    failed.append(name)
                    print(f"failed  {name}: {exc}", file=sys.stderr)

            print(
                f"\n{len(deleted)} deleted, {len(failed)} failed — from "
                f"{args.server}:{args.remote_dir}"
            )
            return 0 if not failed else 2
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
