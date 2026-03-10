"""TinyReadAloud update checker - queries GitHub Releases for new versions."""

import json
import os
import subprocess
import tempfile
import urllib.request
from typing import NamedTuple, Optional

GITHUB_REPO = "dorof/TinyReadAloud"
RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
TIMEOUT = 10


class UpdateInfo(NamedTuple):
    available: bool
    tag: str
    download_url: str
    release_notes: str


def check_for_update(current_version: str) -> Optional[UpdateInfo]:
    """Check GitHub Releases API for a newer version.

    Returns UpdateInfo if check succeeded (available=True/False),
    or None if the check failed (network error, etc.).
    """
    try:
        req = urllib.request.Request(
            RELEASES_API,
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "TinyReadAloud-Updater",
            },
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        tag = data.get("tag_name", "").lstrip("v")
        if not tag:
            return None

        # Compare version tuples
        try:
            remote = tuple(int(x) for x in tag.split("."))
            local = tuple(int(x) for x in current_version.split("."))
        except (ValueError, AttributeError):
            return None

        if remote <= local:
            return UpdateInfo(False, tag, "", "")

        # Find the Setup .exe asset
        download_url = ""
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".exe") and "Setup" in name:
                download_url = asset["browser_download_url"]
                break

        return UpdateInfo(True, tag, download_url, data.get("body", ""))

    except Exception:
        return None


def download_and_run_installer(url: str, on_progress=None) -> bool:
    """Download the installer .exe to a temp dir, launch it, return success."""
    try:
        filename = url.rsplit("/", 1)[-1]
        dest = os.path.join(tempfile.gettempdir(), filename)

        def reporthook(block, block_size, total):
            if on_progress:
                done = block * block_size
                on_progress(done, total)

        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
        # Silent in-place upgrade: close running app, suppress UI, and do not reboot.
        subprocess.Popen(
            [dest, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"],
            shell=False,
        )
        return True
    except Exception:
        return False
