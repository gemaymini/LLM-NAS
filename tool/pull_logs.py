"""Pull matching ``.log`` files from a remote server directory to a local one.

A small helper for the LLM-ENAS-PKB workflow: experiments run on a remote GPU
server write ``search_*.log`` / ``llm_prompts_*.log`` files there, and this tool
copies the ones whose *filename* contains a given substring (e.g. a seed number,
a dataset name, or a hyperparameter tag) down to a local directory for inspection.

It connects over SFTP in a *single* session and downloads every match over that
one connection, so the password is asked at most once.  Connection / auth details
are shared with ``delete_logs.py`` in ``tool/_ssh.py``.

Examples
--------
    # Put the password in .env once (SSH_PASSWORD=secret), then:
    python tool/pull_logs.py --server alice@server1 --remote-dir /home/alice/llm-nas \
        --contains cifar10 --local-dir ./logs

    # A path under the *remote* home dir — quote the ``~`` so your local shell does
    # not expand it; the tool expands it on the server instead.
    python tool/pull_logs.py --server alice@server1 --remote-dir '~/llm-nas' \
        --contains cifar100 --local-dir ./logs

    # Just list what would be transferred (no download)
    python tool/pull_logs.py --server server1 --remote-dir /data/runs \
        --contains seed63980827 --local-dir ./logs --dry-run
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
        prog="pull_logs.py",
        description="Copy remote *.log files whose name contains a substring to a local dir.",
    )
    add_connection_args(p)
    p.add_argument(
        "--contains",
        required=True,
        metavar="SUBSTR",
        help="Only download *.log files whose filename contains this substring",
    )
    p.add_argument(
        "--local-dir",
        default=".",
        metavar="DIR",
        help="Local destination directory (created if missing; default: .)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="List matching files without downloading them"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Re-download files that already exist locally"
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-file output")
    return p.parse_args(argv)


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
                print(f"{len(matching)} matching *.log file(s) on {args.server}:{args.remote_dir}:")
                for path in matching:
                    print(f"  {os.path.basename(path)}")
                return 0

            # Download each matching file, skipping ones already present.
            os.makedirs(args.local_dir, exist_ok=True)
            downloaded: list[str] = []
            skipped: list[str] = []
            failed: list[str] = []

            for path in matching:
                name = os.path.basename(path)
                local_path = os.path.join(args.local_dir, name)
                if os.path.exists(local_path) and not args.overwrite:
                    skipped.append(name)
                    continue
                try:
                    sftp.get(path, local_path)
                    downloaded.append(name)
                    if not args.quiet:
                        print(f"downloaded {name}")
                except OSError as exc:
                    failed.append(name)
                    print(f"failed    {name}: {exc}", file=sys.stderr)

            print(
                f"\n{len(downloaded)} downloaded, {len(skipped)} skipped (already present), "
                f"{len(failed)} failed — into {os.path.abspath(args.local_dir)}"
            )
            return 0 if not failed else 2
        finally:
            sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
