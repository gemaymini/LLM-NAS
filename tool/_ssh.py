"""Shared SSH/SFTP helpers for the ``tool/`` scripts.

``pull_logs.py`` (download) and ``delete_logs.py`` (remove) both talk to a remote
server over a single paramiko SFTP session, authenticate the same way, and list
``*.log`` files in a remote directory.  The connection plumbing lives here so the
two scripts stay DRY.

Authentication, in order of priority:

1. an explicit private key (``--key``), or
2. a password read from the environment / ``.env`` as ``SSH_PASSWORD``
   (``--password`` overrides it), or
3. the default SSH keys / agent.

The scripts are run as ``python tool/<name>.py``, which puts the ``tool/``
directory at the front of ``sys.path``, so ``import _ssh`` resolves to this file.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys

import paramiko
from dotenv import load_dotenv

LOG_EXT = ".log"


def add_connection_args(p: argparse.ArgumentParser) -> None:
    """Add the connection flags shared by every ``tool/`` script."""
    p.add_argument(
        "--server", required=True, metavar="[user@]host", help="SSH target, e.g. alice@server1"
    )
    p.add_argument(
        "--remote-dir",
        required=True,
        metavar="DIR",
        help="Remote directory to operate on (non-recursive; `~` expands on the server)",
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
    load_dotenv()  # make SSH_PASSWORD in .env visible without a manual export
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


def hint_if_local_tilde(remote_dir: str) -> None:
    """Point out the common "forgot to quote ``~``" mistake when it looks applicable."""
    local_home = os.path.expanduser("~")
    if remote_dir == local_home or remote_dir.startswith(local_home + "/"):
        print(
            "hint: the `~` in --remote-dir was expanded by your *local* shell to "
            f"{local_home!r}. Quote it so it expands on the server instead:\n"
            "      --remote-dir '~/path/on/server'",
            file=sys.stderr,
        )
