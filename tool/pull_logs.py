"""Pull matching ``.log`` files from a remote server directory to a local one.

A small helper for the LLM-ENAS-PKB workflow: experiments run on a remote GPU
server write ``search_*.log`` / ``llm_prompts_*.log`` files there, and this tool
copies the ones whose *filename* contains a given substring (e.g. a seed number,
a dataset name, or a hyperparameter tag) down to a local directory for inspection.

It connects over SFTP (paramiko) in a *single* session and downloads every match
over that one connection, so the password is asked at most once.  Authentication,
in order of priority:

1. an explicit private key (``--key``), or
2. a password read from the environment / ``.env`` as ``SSH_PASSWORD``
   (``--password`` overrides it), or
3. the default SSH keys / agent.

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
import stat
import sys

import paramiko
from dotenv import load_dotenv

LOG_EXT = ".log"


def resolve_password(args: argparse.Namespace) -> str | None:
    """Return the SSH password: CLI ``--password`` first, else the ``SSH_PASSWORD`` env var."""
    return args.password or os.environ.get("SSH_PASSWORD")


def split_server(server: str) -> tuple[str | None, str]:
    """Split ``[user@]host`` into ``(username, host)``; username is None if absent."""
    if "@" in server:
        user, host = server.rsplit("@", 1)
        return user or None, host
    return None, server


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    """Open one SSH session, authenticating per the priority in the module docstring."""
    username, host = split_server(args.server)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict = {
        "hostname": host,
        "port": args.port or 22,
        "username": username,
        "timeout": 15,
        "look_for_keys": True,
        "allow_agent": True,
    }
    if args.key:
        connect_kwargs["key_filename"] = args.key
    password = resolve_password(args)
    if password:
        # A password is available: authenticate with it directly and skip key/agent
        # lookups, so a stray mismatched key cannot consume auth attempts first.
        connect_kwargs.update(password=password, look_for_keys=False, allow_agent=False)
    try:
        client.connect(**connect_kwargs)
    except paramiko.AuthenticationException as exc:
        print(
            "error: authentication failed — set SSH_PASSWORD in .env, or pass "
            f"--password/--key. ({exc})",
            file=sys.stderr,
        )
        sys.exit(1)
    return client


def remote_home(client: paramiko.SSHClient) -> str:
    """Return the remote user's home directory (``$HOME`` expanded on the server)."""
    _stdin, stdout, _stderr = client.exec_command("echo $HOME")
    return stdout.read().decode().strip()


def expand_remote_dir(client: paramiko.SSHClient, remote_dir: str) -> str:
    """Resolve a leading ``~`` against the *remote* home directory."""
    remote_dir = remote_dir.rstrip("/")
    if remote_dir == "~":
        return remote_home(client)
    if remote_dir.startswith("~/"):
        return remote_home(client) + remote_dir[1:]
    return remote_dir


def list_remote_logs(sftp: paramiko.SFTPClient, remote_dir: str) -> list[str]:
    """Return the full remote paths of every ``*.log`` file directly under ``remote_dir``."""
    entries = sftp.listdir_attr(remote_dir)
    return [
        os.path.join(remote_dir, entry.filename)
        for entry in entries
        if stat.S_ISREG(entry.st_mode) and entry.filename.endswith(LOG_EXT)
    ]


def _hint_if_local_tilde(remote_dir: str) -> None:
    """Point out the common "forgot to quote ``~``" mistake when it looks applicable."""
    local_home = os.path.expanduser("~")
    if remote_dir == local_home or remote_dir.startswith(local_home + "/"):
        print(
            "hint: the `~` in --remote-dir was expanded by your *local* shell to "
            f"{local_home!r}. Quote it so it expands on the server instead:\n"
            "      --remote-dir '~/path/on/server'",
            file=sys.stderr,
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pull_logs.py",
        description="Copy remote *.log files whose name contains a substring to a local dir.",
    )
    p.add_argument(
        "--server",
        required=True,
        metavar="[user@]host",
        help="SSH target, e.g. alice@server1",
    )
    p.add_argument(
        "--remote-dir",
        required=True,
        metavar="DIR",
        help="Remote directory to search (non-recursive; `~` expands on the server)",
    )
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
    p.add_argument("--port", type=int, default=None, help="SSH port (default: 22)")
    p.add_argument(
        "--key",
        default=None,
        metavar="PATH",
        help="SSH private key (default: default keys / agent, or SSH_PASSWORD from .env)",
    )
    p.add_argument(
        "--password",
        default=None,
        help="SSH password (prefer setting SSH_PASSWORD in .env instead)",
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
    load_dotenv()  # read .env so SSH_PASSWORD is available without manual export
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
                _hint_if_local_tilde(args.remote_dir)
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
