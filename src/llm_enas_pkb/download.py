"""Download the NAS-Bench-201 benchmark archive.

The official ``NAS-Bench-201-v1_1-096897.pth`` is distributed via Google Drive.
``gdown`` (an optional dependency) is used to fetch it; if it is unavailable the
script falls back to a plain HTTP attempt and, finally, prints manual
instructions.

Google Drive frequently blocks scripted downloads of large files, so this module
tries three strategies in order — ``gdown``, raw ``requests``, then a helpful
error pointing the user at the manual download page.
"""

from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

# Google Drive file id for NAS-Bench-201-v1_1-096897.pth (official repo,
# "recommended" link).  Note: 1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKsV is the OLD v1.0
# file which only supports CIFAR10-VALID.
_FILE_ID = "16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_"
_FILE_NAME = "NAS-Bench-201-v1_1-096897.pth"


def download_nasbench201(target: str) -> str:
    """Download the archive to ``target`` and return the path, if successful.

    Idempotent: if ``target`` already exists it is returned untouched.  Raises
    ``RuntimeError`` (with manual instructions) if every strategy fails.
    """
    # Ensure the destination directory exists before any download attempt.
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    if os.path.isfile(target):
        return target

    # 1. gdown (reliable for Google Drive).  Imported lazily because it is an
    #    optional dependency not required to run the rest of the pipeline.
    try:
        import gdown

        logger.info("Downloading NAS-Bench-201 via gdown ...")
        gdown.download(id=_FILE_ID, output=target, quiet=False)
        if os.path.isfile(target):
            return target
    except Exception as exc:  # noqa: BLE001
        logger.warning("gdown download failed: %s", exc)

    # 2. Plain requests (may not handle the large-file confirmation page).
    try:
        import requests

        # The ``uc?export=download`` endpoint is the classic direct-link URL; for
        # large files Google instead serves an HTML virus-scan confirmation page,
        # which is why this fallback may silently produce a small HTML file.
        url = f"https://drive.google.com/uc?export=download&id={_FILE_ID}"
        logger.info("Downloading NAS-Bench-201 via requests ...")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(target, "wb") as fh:
                shutil.copyfileobj(resp.raw, fh)
        # A real archive is >1MB; a tiny file means we got the HTML page instead.
        if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
            return target
    except Exception as exc:  # noqa: BLE001
        logger.warning("requests download failed: %s", exc)

    raise RuntimeError(
        "Could not download NAS-Bench-201 automatically (Google Drive usually blocks "
        "scripted downloads of large files). Download it manually:\n"
        f"  1. Visit https://drive.google.com/file/d/{_FILE_ID}/view\n"
        f"  2. Save the file as '{os.path.abspath(target)}'\n"
        "     (or pass its location via --data-path / LLMENAS_DATA_PATH)."
    )


if __name__ == "__main__":
    import sys

    # Allow ``python -m ...download.py [destination]``; defaults to data/<file>.
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", _FILE_NAME)
    logging.basicConfig(level=logging.INFO)
    print(download_nasbench201(target))
